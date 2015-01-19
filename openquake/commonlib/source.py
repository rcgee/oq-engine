# Copyright (c) 2010-2015, GEM Foundation.
#
# OpenQuake is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OpenQuake is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake.  If not, see <http://www.gnu.org/licenses/>.

import logging
import operator
import collections
import random
from lxml import etree

from openquake.baselib.general import AccumDict
from openquake.commonlib.node import read_nodes
from openquake.commonlib import valid, logictree, sourceconverter
from openquake.commonlib.nrml import nodefactory, PARSE_NS_MAP


class DuplicatedID(Exception):
    """Raised when two sources with the same ID are found in a source model"""


LtRealization = collections.namedtuple(
    'LtRealization', 'ordinal sm_lt_path gsim_lt_path weight')


SourceModel = collections.namedtuple(
    'SourceModel', 'name weight path trt_models gsim_lt ordinal')


class TrtModel(collections.Sequence):
    """
    A container for the following parameters:

    :param str trt:
        the tectonic region type all the sources belong to
    :param list sources:
        a list of hazardlib source objects
    :param int num_ruptures:
        the total number of ruptures generated by the given sources
    :param min_mag:
        the minimum magnitude among the given sources
    :param max_mag:
        the maximum magnitude among the given sources
    :param gsims:
        the GSIMs associated to tectonic region type
    :param id:
        an optional numeric ID (default None) useful to associate
        the model to a database object
    """
    POINT_SOURCE_WEIGHT = 1 / 40.

    def __init__(self, trt, sources=None, num_ruptures=0,
                 min_mag=None, max_mag=None, gsims=None, id=0):
        self.trt = trt
        self.sources = sources or []
        self.num_ruptures = num_ruptures
        self.min_mag = min_mag
        self.max_mag = max_mag
        self.gsims = gsims or []
        self.id = id
        for src in self.sources:
            self.update(src)

    def update(self, src):
        """
        Update the attributes sources, min_mag, max_mag
        according to the given source.

        :param src:
            an instance of :class:
            `openquake.hazardlib.source.base.BaseSeismicSource`
        """
        assert src.tectonic_region_type == self.trt, (
            src.tectonic_region_type, self.trt)
        self.sources.append(src)
        min_mag, max_mag = src.get_min_max_mag()
        prev_min_mag = self.min_mag
        if prev_min_mag is None or min_mag < prev_min_mag:
            self.min_mag = min_mag
        prev_max_mag = self.max_mag
        if prev_max_mag is None or max_mag > prev_max_mag:
            self.max_mag = max_mag

    def update_num_ruptures(self, src):
        """
        Update the attribute num_ruptures according to the given source.

        :param src:
            an instance of :class:
            `openquake.hazardlib.source.base.BaseSeismicSource`
        :returns:
            the weight of the source, as a function of the number
            of ruptures generated by the source
        """
        num_ruptures = src.count_ruptures()
        self.num_ruptures += num_ruptures
        weight = (num_ruptures * self.POINT_SOURCE_WEIGHT
                  if src.__class__.__name__ == 'PointSource'
                  else num_ruptures)
        return weight

    def split_sources_and_count_ruptures(self, area_source_discretization):
        """
        Split the current .sources and replace them with new ones.
        Also, update the total .num_ruptures and the .weigth of each
        source. Finally, make sure the sources are ordered.

        :param area_source_discretization: parameter from the job.ini
        """
        sources = []
        for src in self:
            for ss in sourceconverter.split_source(
                    src, area_source_discretization):
                ss.weight = self.update_num_ruptures(ss)
                sources.append(ss)
        self.sources = sorted(sources, key=operator.attrgetter('source_id'))

    def __repr__(self):
        return '<%s #%d %s, %d source(s)>' % (
            self.__class__.__name__, self.id, self.trt, len(self.sources))

    def __lt__(self, other):
        """
        Make sure there is a precise ordering of TrtModel objects.
        Objects with less sources are put first; in case the number
        of sources is the same, use lexicographic ordering on the trts
        """
        num_sources = len(self.sources)
        other_sources = len(other.sources)
        if num_sources == other_sources:
            return self.trt < other.trt
        return num_sources < other_sources

    def __getitem__(self, i):
        return self.sources[i]

    def __iter__(self):
        return iter(self.sources)

    def __len__(self):
        return len(self.sources)


def parse_source_model(fname, converter, apply_uncertainties=lambda src: None):
    """
    Parse a NRML source model and return an ordered list of TrtModel
    instances.

    :param str fname:
        the full pathname of the source model file
    :param converter:
        :class:`openquake.commonlib.source.SourceConverter` instance
    :param apply_uncertainties:
        a function modifying the sources (or do nothing)
    """
    converter.fname = fname
    source_stats_dict = {}
    source_ids = set()
    src_nodes = read_nodes(fname, lambda elem: 'Source' in elem.tag,
                           nodefactory['sourceModel'])
    for no, src_node in enumerate(src_nodes, 1):
        src = converter.convert_node(src_node)
        if src.source_id in source_ids:
            raise DuplicatedID(
                'The source ID %s is duplicated!' % src.source_id)
        apply_uncertainties(src)
        trt = src.tectonic_region_type
        if trt not in source_stats_dict:
            source_stats_dict[trt] = TrtModel(trt)
        source_stats_dict[trt].update(src)
        source_ids.add(src.source_id)
        if no % 10000 == 0:  # log every 10,000 sources parsed
            logging.info('Parsed %d sources from %s', no, fname)

    # return ordered TrtModels
    return sorted(source_stats_dict.itervalues())


def agg_prob(acc, prob):
    """Aggregation function for probabilities"""
    return 1. - (1. - acc) * (1. - prob)


class RlzsAssoc(collections.Mapping):
    """
    Realization association class. It should not be instantiated directly,
    but only via the method :meth:
    `openquake.commonlib.source.CompositeSourceModel.get_rlzs_assoc`.

    :attr realizations: list of LtRealization objects
    :attr gsim_by_trt: list of dictionaries {trt: gsim}
    :attr rlzs_assoc: dictionary {trt_model_id, gsim: rlzs}

    For instance, for the non-trivial logic tree in
    :mod:`openquake.qa_tests_data.classical.case_15`, which has 4 tectonic
    region types and 4 + 2 + 2 realizations, there are the following
    associations:

    (0, 'BooreAtkinson2008') ['#0-SM1-BA2008_C2003', '#1-SM1-BA2008_T2002']
    (0, 'CampbellBozorgnia2008') ['#2-SM1-CB2008_C2003', '#3-SM1-CB2008_T2002']
    (1, 'Campbell2003') ['#0-SM1-BA2008_C2003', '#2-SM1-CB2008_C2003']
    (1, 'ToroEtAl2002') ['#1-SM1-BA2008_T2002', '#3-SM1-CB2008_T2002']
    (2, 'BooreAtkinson2008') ['#4-SM2_a3pt2b0pt8-BA2008']
    (2, 'CampbellBozorgnia2008') ['#5-SM2_a3pt2b0pt8-CB2008']
    (3, 'BooreAtkinson2008') ['#6-SM2_a3b1-BA2008']
    (3, 'CampbellBozorgnia2008') ['#7-SM2_a3b1-CB2008']
    """
    def __init__(self, rlzs_assoc=None):
        self.realizations = []
        self.gsim_by_trt = []  # [trt -> gsim]
        self.rlzs_assoc = rlzs_assoc or collections.defaultdict(list)

    def _add_realizations(self, idx, lt_model, realizations):
        # create the realizations for the given lt source model
        trt_models = [tm for tm in lt_model.trt_models if tm.num_ruptures]
        if not trt_models:
            return idx
        gsims_by_trt = lt_model.gsim_lt.values
        for gsim_by_trt, weight, gsim_path, _ in realizations:
            weight = float(lt_model.weight) * float(weight)
            rlz = LtRealization(idx, lt_model.path, gsim_path, weight)
            self.realizations.append(rlz)
            self.gsim_by_trt.append(gsim_by_trt)
            for trt_model in trt_models:
                trt = trt_model.trt
                gsim = gsim_by_trt[trt]
                self.rlzs_assoc[trt_model.id, gsim].append(rlz)
                trt_model.gsims = gsims_by_trt[trt]
            idx += 1
        return idx

    def get_gsims_by_trt_id(self):
        """
        Return a dictionary trt_model_id -> [GSIM instances]
        """
        gsims_by_trt = collections.defaultdict(list)
        for trt_id, gsim in sorted(self.rlzs_assoc):
            gsims_by_trt[trt_id].append(valid.gsim(gsim))
        return gsims_by_trt

    def combine(self, results, agg=agg_prob):
        """
        :param results: dictionary (trt_model_id, gsim_name) -> <AccumDict>
        :param agg: aggregation function (default composition of probabilities)
        :returns: a dictionary rlz -> aggregate <AccumDict>

        Example: a case with tectonic region type T1 with GSIMS A, B, C
        and tectonic region type T2 with GSIMS D, E.

        >>> assoc = RlzsAssoc({
        ... ('T1', 'A'): ['r0', 'r1'],
        ... ('T1', 'B'): ['r2', 'r3'],
        ... ('T1', 'C'): ['r4', 'r5'],
        ... ('T2', 'D'): ['r0', 'r2', 'r4'],
        ... ('T2', 'E'): ['r1', 'r3', 'r5']})
        ...
        >>> results = {
        ... ('T1', 'A'): 0.01,
        ... ('T1', 'B'): 0.02,
        ... ('T1', 'C'): 0.03,
        ... ('T2', 'D'): 0.04,
        ... ('T2', 'E'): 0.05,}
        ...
        >>> combinations = assoc.combine(results, operator.add)
        >>> for key, value in sorted(combinations.items()): print key, value
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

        In reality, the `combine` method is used with dictionaries with the
        hazard curves keyed by intensity measure type and the aggregation
        function is the composition of probability, which however is closer
        to the sum for small probabilities.
        """
        acc = 0
        for key, value in results.iteritems():
            for rlz in self.rlzs_assoc[key]:
                acc = agg(acc, AccumDict({rlz: value}))
        return acc

    def collect_by_rlz(self, dicts):
        """
        :param dicts: a list of dicts with key (trt_model_id, gsim)
        :returns: a dictionary of lists keyed by realization
        """
        dicts_by_rlz = AccumDict()  # rlz -> list
        for dic in dicts:
            items = self.combine(dic).iteritems()
            dicts_by_rlz += {rlz: [val] for rlz, val in items}
        return dicts_by_rlz

    def __iter__(self):
        return self.rlzs_assoc.iterkeys()

    def __getitem__(self, key):
        return self.rlzs_assoc[key]

    def __len__(self):
        return len(self.rlzs_assoc)


class CompositeSourceModel(collections.Sequence):
    """
    :param source_model_lt:
        a :class:`openquake.commonlib.readinput.SourceModelLogicTree` instance
    :param source_models:
        a list of :class:`openquake.commonlib.readinput.SourceModel` tuples
    """
    def __init__(self, source_model_lt, source_models):
        self.source_model_lt = source_model_lt
        self.source_models = list(source_models)
        if not self.source_models:
            raise RuntimeError('All sources were filtered away')
        self.tmdict = {}
        for i, tm in enumerate(self.trt_models):
            tm.id = i
            self.tmdict[i] = tm

    @property
    def trt_models(self):
        """
        Yields the TrtModels inside each source model.
        """
        for sm in self.source_models:
            for trt_model in sm.trt_models:
                yield trt_model

    @property
    def sources(self):
        """
        Yield the sources contained in the internal source models.
        """
        for trt_model in self.trt_models:
            for src in trt_model:
                src.trt_model_id = trt_model.id
                yield src

    def reduce_trt_models(self):
        """
        Remove the tectonic regions without ruptures and reduce the
        GSIM logic tree. It works by updating the underlying source models.
        """
        for sm in self:
            trts = set(trt_model.trt for trt_model in sm.trt_models
                       if trt_model.num_ruptures > 0)
            if trts == set(sm.gsim_lt.tectonic_region_types):
                # nothing to remove
                continue
            # build the reduced logic tree
            gsim_lt = sm.gsim_lt.filter(trts)
            tmodels = []  # collect the reduced trt models
            for trt_model in sm.trt_models:
                if trt_model.trt in trts:
                    trt_model.gsims = gsim_lt.values[trt_model.trt]
                    tmodels.append(trt_model)
            self[sm.ordinal] = SourceModel(
                sm.name, sm.weight, sm.path, tmodels, gsim_lt, sm.ordinal)

    def get_source_model(self, path):
        """
        Extract a specific source model, given its logic tree path.

        :param path: the source model logic tree path as a tuple of string
        """
        for sm in self:
            if sm.path == path:
                return sm
        raise KeyError(
            'There is no source model with sm_lt_path=%s' % str(path))

    def get_rlzs_assoc(self):
        """
        Return a RlzsAssoc with fields realizations, gsim_by_trt,
        rlz_idx and trt_gsims.
        """
        assoc = RlzsAssoc()
        random_seed = self.source_model_lt.seed
        num_samples = self.source_model_lt.num_samples
        idx = 0
        for sm_name, weight, sm_lt_path, _ in self.source_model_lt:
            lt_model = self.get_source_model(sm_lt_path)
            if num_samples:  # sampling, pick just one gsim realization
                rnd = random.Random(random_seed + idx)
                rlzs = [logictree.sample_one(lt_model.gsim_lt, rnd)]
            else:
                rlzs = list(lt_model.gsim_lt)  # full enumeration
            logging.info('Creating %d GMPE realization(s) for model %s, %s',
                         len(rlzs), lt_model.name, lt_model.path)
            idx = assoc._add_realizations(idx, lt_model, rlzs)

        # TODO: if num_samples > total_num_paths we should add a warning here,
        # see https://bugs.launchpad.net/oq-engine/+bug/1367273
        return assoc

    def __getitem__(self, i):
        """Return the i-th source model"""
        return self.source_models[i]

    def __setitem__(self, i, sm):
        """Update the i-th source model"""
        self.source_models[i] = sm

    def __iter__(self):
        """Return an iterator over the underlying source models"""
        return iter(self.source_models)

    def __len__(self):
        """Return the number of underlying source models"""
        return len(self.source_models)


def _collect_source_model_paths(smlt):
    """
    Given a path to a source model logic tree or a file-like, collect all of
    the soft-linked path names to the source models it contains and return them
    as a uniquified list (no duplicates).
    """
    src_paths = []
    tree = etree.parse(smlt)
    for branch_set in tree.xpath('//nrml:logicTreeBranchSet',
                                 namespaces=PARSE_NS_MAP):

        if branch_set.get('uncertaintyType') == 'sourceModel':
            for branch in branch_set.xpath(
                    './nrml:logicTreeBranch/nrml:uncertaintyModel',
                    namespaces=PARSE_NS_MAP):
                src_paths.append(branch.text)
    return sorted(set(src_paths))
