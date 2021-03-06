# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2015-2016 GEM Foundation
#
# OpenQuake is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OpenQuake is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake. If not, see <http://www.gnu.org/licenses/>.

import operator
import logging
import collections
import numpy

from openquake.baselib import hdf5
from openquake.baselib.python3compat import zip
from openquake.baselib.performance import Monitor
from openquake.baselib.general import (
    groupby, split_in_blocks, get_array)
from openquake.hazardlib import site, calc
from openquake.risklib import scientific, riskmodels

U8 = numpy.uint8
U16 = numpy.uint16
U32 = numpy.uint32
F32 = numpy.float32

FIELDS = ('site_id', 'lon', 'lat', 'idx', 'taxonomy_id', 'area', 'number',
          'occupants', 'deductible-', 'insurance_limit-', 'retrofitted-')

by_taxonomy = operator.attrgetter('taxonomy')


def agg_prob(acc, prob):
    """Aggregation function for probabilities"""
    return 1. - (1. - acc) * (1. - prob)


def combine(rlzs_assoc, results, agg=agg_prob):
    """
    :param results: a dictionary (src_group_id, gsim) -> floats
    :param agg: an aggregation function
    :returns: a dictionary rlz -> aggregated floats

    Example: a case with tectonic region type T1 with GSIMS A, B, C
    and tectonic region type T2 with GSIMS D, E.

    >> assoc = RlzsAssoc(CompositionInfo([], []))
    >> assoc.rlzs_assoc = {
    ... ('T1', 'A'): ['r0', 'r1'],
    ... ('T1', 'B'): ['r2', 'r3'],
    ... ('T1', 'C'): ['r4', 'r5'],
    ... ('T2', 'D'): ['r0', 'r2', 'r4'],
    ... ('T2', 'E'): ['r1', 'r3', 'r5']}
    ...
    >> results = {
    ... ('T1', 'A'): 0.01,
    ... ('T1', 'B'): 0.02,
    ... ('T1', 'C'): 0.03,
    ... ('T2', 'D'): 0.04,
    ... ('T2', 'E'): 0.05,}
    ...
    >> combinations = combine(assoc, results, operator.add)
    >> for key, value in sorted(combinations.items()): print key, value
    r0 0.05
    r1 0.06
    r2 0.06
    r3 0.07
    r4 0.07
    r5 0.08

    You can check that all the possible sums are performed:

    r0: 0.01 + 0.04 (T1A + T2D)
    r1: 0.01 + 0.05 (T1A + T2E)
    r2: 0.02 + 0.04 (T1B + T2D)
    r3: 0.02 + 0.05 (T1B + T2E)
    r4: 0.03 + 0.04 (T1C + T2D)
    r5: 0.03 + 0.05 (T1C + T2E)

    In reality, the `.combine` function is used with hazard_curves and
    the aggregation function is the `agg_curves` function, a composition of
    probability, which however is close to the sum for small probabilities.
    """
    ad = {rlz: 0 for rlz in rlzs_assoc.realizations}
    for key, value in results.items():
        for rlz in rlzs_assoc.rlzs_assoc[key]:
            ad[rlz] = agg(ad[rlz], value)
    return ad


class AssetCollection(object):
    D, I, R = len('deductible-'), len('insurance_limit-'), len('retrofitted-')

    def __init__(self, assets_by_site, cost_calculator, time_event,
                 time_events=''):
        self.cc = cost_calculator
        self.time_event = time_event
        self.time_events = hdf5.array_of_vstr(time_events)
        self.array, self.taxonomies = self.build_asset_collection(
            assets_by_site, time_event)
        fields = self.array.dtype.names
        self.loss_types = sorted(
            f[6:] for f in fields if f.startswith('value-'))
        self.deduc = [n for n in fields if n.startswith('deductible-')]
        self.i_lim = [n for n in fields if n.startswith('insurance_limit-')]
        self.retro = [n for n in fields if n.startswith('retrofitted-')]

    def assets_by_site(self):
        """
        :returns: numpy array of lists with the assets by each site
        """
        assetcol = self.array
        site_ids = sorted(set(assetcol['site_id']))
        assets_by_site = [[] for sid in site_ids]
        index = dict(zip(site_ids, range(len(site_ids))))
        for i, ass in enumerate(assetcol):
            assets_by_site[index[ass['site_id']]].append(self[i])
        return numpy.array(assets_by_site)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, indices):
        if isinstance(indices, int):  # single asset
            a = self.array[indices]
            values = {lt: a['value-' + lt] for lt in self.loss_types}
            if 'occupants' in self.array.dtype.names:
                values['occupants_' + str(self.time_event)] = a['occupants']
            return riskmodels.Asset(
                    a['idx'],
                    self.taxonomies[a['taxonomy_id']],
                    number=a['number'],
                    location=(a['lon'], a['lat']),
                    values=values,
                    area=a['area'],
                    deductibles={lt[self.D:]: a[lt] for lt in self.deduc},
                    insurance_limits={lt[self.I:]: a[lt] for lt in self.i_lim},
                    retrofitteds={lt[self.R:]: a[lt] for lt in self.retro},
                    calc=self.cc, ordinal=indices)
        new = object.__new__(self.__class__)
        new.time_event = self.time_event
        new.array = self.array[indices]
        new.taxonomies = self.taxonomies
        return new

    def __len__(self):
        return len(self.array)

    def __toh5__(self):
        attrs = {'time_event': self.time_event or 'None',
                 'time_events': self.time_events,
                 'loss_types': hdf5.array_of_vstr(self.loss_types),
                 'deduc': hdf5.array_of_vstr(self.deduc),
                 'i_lim': hdf5.array_of_vstr(self.i_lim),
                 'retro': hdf5.array_of_vstr(self.retro),
                 'nbytes': self.array.nbytes}
        return dict(array=self.array, taxonomies=self.taxonomies,
                    cost_calculator=self.cc), attrs

    def __fromh5__(self, dic, attrs):
        vars(self).update(attrs)
        self.array = dic['array'].value
        self.taxonomies = dic['taxonomies'].value
        self.cc = dic['cost_calculator']

    @staticmethod
    def build_asset_collection(assets_by_site, time_event=None):
        """
        :param assets_by_site: a list of lists of assets
        :param time_event: a time event string (or None)
        :returns: two arrays `assetcol` and `taxonomies`
        """
        for assets in assets_by_site:
            if len(assets):
                first_asset = assets[0]
                break
        else:  # no break
            raise ValueError('There are no assets!')
        candidate_loss_types = list(first_asset.values)
        loss_types = []
        the_occupants = 'occupants_%s' % time_event
        for candidate in candidate_loss_types:
            if candidate.startswith('occupants'):
                if candidate == the_occupants:
                    loss_types.append('occupants')
                # discard occupants for different time periods
            else:
                loss_types.append('value-' + candidate)
        deductible_d = first_asset.deductibles or {}
        limit_d = first_asset.insurance_limits or {}
        retrofitting_d = first_asset.retrofitteds or {}
        deductibles = ['deductible-%s' % name for name in deductible_d]
        limits = ['insurance_limit-%s' % name for name in limit_d]
        retrofittings = ['retrofitted-%s' % n for n in retrofitting_d]
        float_fields = loss_types + deductibles + limits + retrofittings
        taxonomies = set()
        for assets in assets_by_site:
            for asset in assets:
                taxonomies.add(asset.taxonomy)
        sorted_taxonomies = sorted(taxonomies)
        asset_dt = numpy.dtype(
            [('idx', U32), ('lon', F32), ('lat', F32), ('site_id', U32),
             ('taxonomy_id', U32), ('number', F32), ('area', F32)] + [
                 (str(name), float) for name in float_fields])
        num_assets = sum(len(assets) for assets in assets_by_site)
        assetcol = numpy.zeros(num_assets, asset_dt)
        asset_ordinal = 0
        fields = set(asset_dt.fields)
        for sid, assets_ in enumerate(assets_by_site):
            for asset in sorted(assets_, key=operator.attrgetter('id')):
                asset.ordinal = asset_ordinal
                record = assetcol[asset_ordinal]
                asset_ordinal += 1
                for field in fields:
                    if field == 'taxonomy_id':
                        value = sorted_taxonomies.index(asset.taxonomy)
                    elif field == 'number':
                        value = asset.number
                    elif field == 'area':
                        value = asset.area
                    elif field == 'idx':
                        value = asset.id
                    elif field == 'site_id':
                        value = sid
                    elif field == 'lon':
                        value = asset.location[0]
                    elif field == 'lat':
                        value = asset.location[1]
                    elif field == 'occupants':
                        value = asset.values[the_occupants]
                    else:
                        try:
                            name, lt = field.split('-')
                        except ValueError:  # no - in field
                            name, lt = 'value', field
                        # the line below retrieve one of `deductibles`,
                        # `insured_limits` or `retrofitteds` ("s" suffix)
                        value = getattr(asset, name + 's')[lt]
                    record[field] = value
        return assetcol, numpy.array(sorted_taxonomies, hdf5.vstr)


class CompositeRiskModel(collections.Mapping):
    """
    A container (imt, taxonomy) -> riskmodel.

    :param riskmodels: a dictionary (imt, taxonomy) -> riskmodel
    :param damage_states: None or a list of damage states
    """
    def __init__(self, riskmodels, damage_states=None):
        self.damage_states = damage_states  # not None for damage calculations
        self._riskmodels = riskmodels
        self.loss_types = []
        self.curve_builders = []
        self.lti = {}  # loss_type -> idx
        self.covs = 0  # number of coefficients of variation
        self.taxonomies = []  # populated in get_risk_model

    def get_min_iml(self):
        iml = collections.defaultdict(list)
        for taxo, rm in self._riskmodels.items():
            for lt, rf in rm.risk_functions.items():
                iml[rf.imt].append(rf.imls[0])
        return {imt: min(iml[imt]) for imt in iml}

    def build_loss_dtypes(self, conditional_loss_poes, insured_losses=False):
        """
        :param conditional_loss_poes:
            configuration parameter
        :param insured_losses:
            configuration parameter
        :returns:
           loss_curve_dt and loss_maps_dt
        """
        lst = [('poe-%s' % poe, F32) for poe in conditional_loss_poes]
        if insured_losses:
            lst += [(name + '_ins', pair) for name, pair in lst]
        lm_dt = numpy.dtype(lst)
        lc_list = []
        lm_list = []
        for cb in (b for b in self.curve_builders if b.user_provided):
            pairs = [('losses', (F32, cb.curve_resolution)),
                     ('poes', (F32, cb.curve_resolution)),
                     ('avg', F32)]
            if insured_losses:
                pairs += [(name + '_ins', pair) for name, pair in pairs]
            lc_list.append((cb.loss_type, numpy.dtype(pairs)))
            lm_list.append((cb.loss_type, lm_dt))
        loss_curve_dt = numpy.dtype(lc_list) if lc_list else None
        loss_maps_dt = numpy.dtype(lm_list) if lm_list else None
        return loss_curve_dt, loss_maps_dt

    # FIXME: scheduled for removal once we change agg_curve to be built from
    # the user-provided loss ratios
    def build_all_loss_dtypes(self, curve_resolution, conditional_loss_poes,
                              insured_losses=False):
        """
        :param conditional_loss_poes:
            configuration parameter
        :param insured_losses:
            configuration parameter
        :returns:
           loss_curve_dt and loss_maps_dt
        """
        lst = [('poe-%s' % poe, F32) for poe in conditional_loss_poes]
        if insured_losses:
            lst += [(name + '_ins', pair) for name, pair in lst]
        lm_dt = numpy.dtype(lst)
        lc_list = []
        lm_list = []
        for loss_type in self.loss_types:
            pairs = [('losses', (F32, curve_resolution)),
                     ('poes', (F32, curve_resolution)),
                     ('avg', F32)]
            if insured_losses:
                pairs += [(name + '_ins', pair) for name, pair in pairs]
            lc_list.append((loss_type, numpy.dtype(pairs)))
            lm_list.append((loss_type, lm_dt))
        loss_curve_dt = numpy.dtype(lc_list) if lc_list else None
        loss_maps_dt = numpy.dtype(lm_list) if lm_list else None
        return loss_curve_dt, loss_maps_dt

    def make_curve_builders(self, oqparam):
        """
        Populate the inner lists .loss_types, .curve_builders.
        """
        default_loss_ratios = numpy.linspace(
            0, 1, oqparam.loss_curve_resolution + 1)[1:]
        loss_types = self._get_loss_types()
        for l, loss_type in enumerate(loss_types):
            if oqparam.calculation_mode in ('classical', 'classical_risk'):
                curve_resolutions = set()
                lines = []
                for key in sorted(self):
                    rm = self[key]
                    if loss_type in rm.loss_ratios:
                        ratios = rm.loss_ratios[loss_type]
                        curve_resolutions.add(len(ratios))
                        lines.append('%s %d' % (
                            rm.risk_functions[loss_type], len(ratios)))
                if len(curve_resolutions) > 1:
                    logging.info(
                        'Different num_loss_ratios:\n%s', '\n'.join(lines))
                cb = scientific.CurveBuilder(
                    loss_type, ratios, True,
                    oqparam.conditional_loss_poes, oqparam.insured_losses,
                    curve_resolution=max(curve_resolutions))
            elif loss_type in oqparam.loss_ratios:  # loss_ratios provided
                cb = scientific.CurveBuilder(
                    loss_type, oqparam.loss_ratios[loss_type], True,
                    oqparam.conditional_loss_poes, oqparam.insured_losses)
            else:  # no loss_ratios provided
                cb = scientific.CurveBuilder(
                    loss_type, default_loss_ratios, False,
                    oqparam.conditional_loss_poes, oqparam.insured_losses)
            self.curve_builders.append(cb)
            self.loss_types.append(loss_type)
            self.lti[loss_type] = l

    def get_loss_ratios(self):
        """
        :returns: a 1-dimensional composite array with loss ratios by loss type
        """
        lst = [('user_provided', numpy.bool)]
        for cb in self.curve_builders:
            lst.append((cb.loss_type, F32, len(cb.ratios)))
        loss_ratios = numpy.zeros(1, numpy.dtype(lst))
        for cb in self.curve_builders:
            loss_ratios['user_provided'] = cb.user_provided
            loss_ratios[cb.loss_type] = tuple(cb.ratios)
        return loss_ratios

    def _get_loss_types(self):
        """
        :returns: a sorted list with all the loss_types contained in the model
        """
        ltypes = set()
        for rm in self.values():
            ltypes.update(rm.loss_types)
        return sorted(ltypes)

    def __getitem__(self, taxonomy):
        return self._riskmodels[taxonomy]

    def __iter__(self):
        return iter(sorted(self._riskmodels))

    def __len__(self):
        return len(self._riskmodels)

    def build_input(self, hazards_by_site, assetcol, eps_dict):
        """
        :param hazards_by_site: an array of hazards per each site
        :param assetcol: AssetCollection instance
        :param eps_dict: a dictionary of epsilons
        :returns: a :class:`RiskInput` instance
        """
        return RiskInput(hazards_by_site, assetcol, eps_dict)

    def build_inputs_from_ruptures(
            self, grp_trt, imts, sitecol, all_ruptures, trunc_level,
            correl_model, min_iml, eps, hint):
        """
        :param imts: list of intensity measure type strings
        :param sitecol: a SiteCollection instance
        :param all_ruptures: the complete list of EBRupture instances
        :param trunc_level: the truncation level (or None)
        :param correl_model: the correlation model (or None)
        :param min_iml: an array of minimum IMLs per IMT
        :param eps: a matrix of epsilons of shape (N, E) or None
        :param hint: hint for how many blocks to generate

        Yield :class:`RiskInputFromRuptures` instances.
        """
        by_grp_id = operator.attrgetter('grp_id')
        start = 0
        for ses_ruptures in split_in_blocks(
                all_ruptures, hint or 1, key=by_grp_id,
                weight=operator.attrgetter('weight')):
            grp_id = ses_ruptures[0].grp_id
            num_events = sum(sr.multiplicity for sr in ses_ruptures)
            idxs = numpy.arange(start, start + num_events)
            start += num_events
            yield RiskInputFromRuptures(
                grp_trt[grp_id], imts, sitecol, ses_ruptures,
                trunc_level, correl_model, min_iml,
                eps[:, idxs] if eps is not None else None)

    def gen_outputs(self, riskinput, rlzs_assoc, monitor,
                    assetcol=None):
        """
        Group the assets per taxonomy and compute the outputs by using the
        underlying riskmodels. Yield the outputs generated as dictionaries
        out_by_lr.

        :param riskinput: a RiskInput instance
        :param rlzs_assoc: a RlzsAssoc instance
        :param monitor: a monitor object used to measure the performance
        :param assetcol: not None only for event based risk
        """
        mon_context = monitor('building context')
        mon_hazard = monitor('building hazard')
        mon_risk = monitor('computing risk', measuremem=False)
        with mon_context:
            assets_by_site = (riskinput.assets_by_site if assetcol is None
                              else assetcol.assets_by_site())
            hazard_getter = riskinput.hazard_getter(
                rlzs_assoc, mon_hazard(measuremem=False))

        # group the assets by taxonomy
        taxonomies = set()
        with monitor('grouping assets by taxonomy'):
            dic = collections.defaultdict(list)
            for i, assets in enumerate(assets_by_site):
                group = groupby(assets, by_taxonomy)
                for taxonomy in group:
                    epsgetter = riskinput.epsilon_getter(
                        [asset.ordinal for asset in group[taxonomy]])
                    dic[taxonomy].append((i, group[taxonomy], epsgetter))
                    taxonomies.add(taxonomy)

        if hasattr(riskinput, 'ses_ruptures'):  # event based
            grp_id = riskinput.ses_ruptures[0].grp_id
            rlzs = rlzs_assoc.get_rlzs_by_grp_id()[grp_id]
        else:
            rlzs = rlzs_assoc.realizations
        for rlz in rlzs:
            with mon_hazard:
                hazard = hazard_getter(rlz)
            for taxonomy in sorted(taxonomies):
                riskmodel = self[taxonomy]
                for lt in self.loss_types:
                    imt = riskmodel.risk_functions[lt].imt
                    with mon_risk:
                        for i, assets, epsgetter in dic[taxonomy]:
                            haz = hazard[i].get(imt, ())
                            if len(haz):
                                out = riskmodel(lt, assets, haz, epsgetter)
                                if out:  # can be None in scenario_risk
                                    out.lr = self.lti[lt], rlz.ordinal
                                    yield out
        if hasattr(hazard_getter, 'gmfbytes'):  # for event based risk
            monitor.gmfbytes = hazard_getter.gmfbytes

    def __repr__(self):
        lines = ['%s: %s' % item for item in sorted(self.items())]
        return '<%s(%d, %d)\n%s>' % (
            self.__class__.__name__, len(lines), self.covs, '\n'.join(lines))


class PoeGetter(object):
    """
    Callable returning a list of N dictionaries imt -> curve
    when called on a realization.
    """
    def __init__(self, hazard_by_site, rlzs_assoc):
        self.rlzs_assoc = rlzs_assoc
        self.rlzs = rlzs_assoc.realizations
        self.hazard_by_site = [{imt: combine(rlzs_assoc, hazard_by_imt[imt])
                                for imt in hazard_by_imt}
                               for hazard_by_imt in hazard_by_site]

    def __call__(self, rlz):
        return [{imt: haz[imt][rlz] for imt in haz}
                for haz in self.hazard_by_site]


class GmfGetter(object):
    """
    Callable returning a list of N dictionaries imt -> array(gmv, eid).
    when called on a realization.
    """
    dt = numpy.dtype([('gmv', F32), ('eid', U32)])

    def __init__(self, gsims, ebruptures, sitecol, imts, min_iml,
                 truncation_level, correlation_model, samples):
        self.gsims = gsims
        self.imts = imts
        self.min_iml = min_iml
        self.truncation_level = truncation_level
        self.correlation_model = correlation_model
        self.samples = samples
        self.sids = sitecol.sids
        self.computers = []
        for ebr in ebruptures:
            sites = site.FilteredSiteCollection(ebr.indices, sitecol.complete)
            computer = calc.gmf.GmfComputer(
                ebr, sites, imts, set(gsims),
                truncation_level, correlation_model)
            self.computers.append(computer)
        self.gmfbytes = 0

    def __call__(self, rlz):
        gsim = self.gsims[rlz.ordinal]
        gmfdict = collections.defaultdict(dict)
        for computer in self.computers:
            rup = computer.rupture
            if self.samples > 1:
                eids = get_array(rup.events, sample=rlz.sampleid)['eid']
            else:
                eids = rup.events['eid']
            array = computer.compute(gsim, len(eids))  # (i, n, e)
            for imti, imt in enumerate(self.imts):
                min_gmv = self.min_iml[imti]
                for eid, gmf in zip(eids, array[imti].T):
                    for sid, gmv in zip(computer.sites.sids, gmf):
                        if gmv > min_gmv:
                            dic = gmfdict[sid]
                            if imt in dic:
                                dic[imt].append((gmv, eid))
                            else:
                                dic[imt] = [(gmv, eid)]
        dicts = []  # a list of dictionaries imt -> array(gmv, eid)
        for sid in self.sids:
            dic = gmfdict[sid]
            for imt in dic:
                dic[imt] = arr = numpy.array(dic[imt], self.dt)
                self.gmfbytes += arr.nbytes
            dicts.append(dic)
        return dicts


class RiskInput(object):
    """
    Contains all the assets and hazard values associated to a given
    imt and site.

    :param imt_taxonomies: a pair (IMT, taxonomies)
    :param hazard_by_site: array of hazards, one per site
    :param assets_by_site: array of assets, one per site
    :param eps_dict: dictionary of epsilons
    """
    def __init__(self, hazard_by_site, assets_by_site, eps_dict):
        self.hazard_by_site = hazard_by_site
        self.assets_by_site = assets_by_site
        self.eps = eps_dict
        taxonomies_set = set()
        aids = []
        for assets in self.assets_by_site:
            for asset in assets:
                taxonomies_set.add(asset.taxonomy)
                aids.append(asset.ordinal)
        self.aids = numpy.array(aids, numpy.uint32)
        self.taxonomies = sorted(taxonomies_set)
        self.eids = None  # for API compatibility with RiskInputFromRuptures
        self.weight = len(self.aids)

    @property
    def imt_taxonomies(self):
        """Return a list of pairs (imt, taxonomies) with a single element"""
        return [(self.imt, self.taxonomies)]

    def epsilon_getter(self, asset_ordinals):
        """
        :param asset_ordinals: list of ordinals of the assets
        :returns: a closure returning an array of epsilons from the event IDs
        """
        return lambda dummy1, dummy2: (
            [self.eps[aid] for aid in asset_ordinals]
            if self.eps else None)

    def hazard_getter(self, rlzs_assoc, monitor=Monitor()):
        """
        :param rlzs_assoc:
            :class:`openquake.commonlib.source.RlzsAssoc` instance
        :param monitor:
            a :class:`openquake.baselib.performance.Monitor` instance
        :returns:
            list of hazard dictionaries imt -> rlz -> haz per each site
        """
        return PoeGetter(self.hazard_by_site, rlzs_assoc)

    def __repr__(self):
        return '<%s taxonomy=%s, %d asset(s)>' % (
            self.__class__.__name__, ', '.join(self.taxonomies), self.weight)


def make_eps(assets_by_site, num_samples, seed, correlation):
    """
    :param assets_by_site: a list of lists of assets
    :param int num_samples: the number of ruptures
    :param int seed: a random seed
    :param float correlation: the correlation coefficient
    :returns: epsilons matrix of shape (num_assets, num_samples)
    """
    all_assets = (a for assets in assets_by_site for a in assets)
    assets_by_taxo = groupby(all_assets, by_taxonomy)
    num_assets = sum(map(len, assets_by_site))
    eps = numpy.zeros((num_assets, num_samples), numpy.float32)
    for taxonomy, assets in assets_by_taxo.items():
        # the association with the epsilons is done in order
        assets.sort(key=operator.attrgetter('id'))
        shape = (len(assets), num_samples)
        logging.info('Building %s epsilons for taxonomy %s', shape, taxonomy)
        zeros = numpy.zeros(shape)
        epsilons = scientific.make_epsilons(zeros, seed, correlation)
        for asset, epsrow in zip(assets, epsilons):
            eps[asset.ordinal] = epsrow
    return eps


class Gmvset(object):
    """
    Emulate a dataset containing ground motion values per event ID,
    realization ordinal and IMT index.
    """
    dt = numpy.dtype([('gmv', F32), ('eid', U32), ('rlzi', U16), ('imti', U8)])

    def __init__(self):
        self.pairs = []

    def append(self, gmv, eid, rlzi, imti):
        self.pairs.append((gmv, eid, rlzi, imti))

    @property
    def value(self):
        return numpy.array(self.pairs, self.dt)

    def __len__(self):
        return len(self.pairs)


def str2rsi(key):
    """
    Convert a string of the form 'rlz-XXXX/sid-YYYY/ZZZ'
    into a triple (XXXX, YYYY, ZZZ)
    """
    rlzi, sid, imt = key.split('/')
    return int(rlzi[4:]), int(sid[4:]), imt


def rsi2str(rlzi, sid, imt):
    """
    Convert a triple (XXXX, YYYY, ZZZ) into a string of the form
    'rlz-XXXX/sid-YYYY/ZZZ'
    """
    return 'rlz-%04d/sid-%04d/%s' % (rlzi, sid, imt)


class RiskInputFromRuptures(object):
    """
    Contains all the assets associated to the given IMT and a subsets of
    the ruptures for a given calculation.

    :param imts: a list of intensity measure type strings
    :param sitecol: SiteCollection instance
    :param assets_by_site: list of list of assets
    :param ses_ruptures: ordered array of EBRuptures
    :param gsims: list of GSIM instances
    :param trunc_level: truncation level for the GSIMs
    :param correl_model: correlation model for the GSIMs
    :param min_iml: an array with the minimum intensity per IMT
    :params epsilons: a matrix of epsilons (or None)
    :param eids: an array of event IDs (or None)
    """
    def __init__(self, trt, imts, sitecol, ses_ruptures,
                 trunc_level, correl_model, min_iml, epsilons=None):
        self.sitecol = sitecol
        self.ses_ruptures = numpy.array(ses_ruptures)
        self.trt = trt
        self.trunc_level = trunc_level
        self.correl_model = correl_model
        self.min_iml = min_iml
        self.weight = sum(sr.weight for sr in ses_ruptures)
        self.imts = imts
        self.eids = numpy.concatenate([r.events['eid'] for r in ses_ruptures])
        if epsilons is not None:
            self.eps = epsilons  # matrix N x E, events in this block
            self.eid2idx = dict(zip(self.eids, range(len(self.eids))))

    def epsilon_getter(self, asset_ordinals):
        """
        :param asset_ordina: ordinal of the asset
        :returns: a closure returning an array of epsilons from the event IDs
        """
        if not hasattr(self, 'eps'):
            return lambda aid, eids: None

        def geteps(aid, eids):
            return self.eps[aid, [self.eid2idx[eid] for eid in eids]]
        return geteps

    def hazard_getter(self, rlzs_assoc, monitor=Monitor()):
        """
        :param rlzs_assoc:
            :class:`openquake.commonlib.source.RlzsAssoc` instance
        :param monitor:
            a :class:`openquake.baselib.performance.Monitor` instance
        :returns:
            lists of N hazard dictionaries imt -> rlz -> Gmvs
        """
        grp_id = self.ses_ruptures[0].grp_id
        gsims = [dic[self.trt] for dic in rlzs_assoc.gsim_by_trt]
        gg = GmfGetter(gsims, self.ses_ruptures, self.sitecol,
                       self.imts, self.min_iml, self.trunc_level,
                       self.correl_model, rlzs_assoc.samples[grp_id])
        return gg

    def __repr__(self):
        return '<%s imts=%s, weight=%d>' % (
            self.__class__.__name__, self.imts, self.weight)
