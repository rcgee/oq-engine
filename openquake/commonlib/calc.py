# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2014-2016 GEM Foundation
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

from __future__ import division
import collections
import itertools
import operator
import logging
import copy

import numpy

from openquake.baselib.general import get_array, group_array, AccumDict
from openquake.hazardlib.imt import from_string
from openquake.hazardlib.calc import filters
from openquake.hazardlib.probability_map import ProbabilityMap
from openquake.hazardlib.site import SiteCollection
from openquake.commonlib import readinput, oqvalidation


MAX_INT = 2 ** 31 - 1  # this is used in the random number generator
# in this way even on 32 bit machines Python will not have to convert
# the generated seed into a long integer

U8 = numpy.uint8
U16 = numpy.uint16
U32 = numpy.uint32
F32 = numpy.float32
F64 = numpy.float64

# ############## utilities for the classical calculator ############### #

SourceRuptureSites = collections.namedtuple(
    'SourceRuptureSites',
    'source rupture sites')


# used in classical and event_based calculators
def combine_pmaps(rlzs_assoc, results):
    """
    :param rlzs_assoc: a :class:`openquake.commonlib.source.RlzsAssoc` instance
    :param results: dictionary src_group_id -> probability map
    :returns: a dictionary rlz -> aggregate probability map
    """
    acc = AccumDict()
    for grp_id in results:
        for i, gsim in enumerate(rlzs_assoc.gsims_by_grp_id[grp_id]):
            pmap = results[grp_id].extract(i)
            for rlz in rlzs_assoc.rlzs_assoc[grp_id, gsim]:
                if rlz in acc:
                    acc[rlz] |= pmap
                else:
                    acc[rlz] = copy.copy(pmap)
    return acc

# ######################### hazard maps ################################### #

# cutoff value for the poe
EPSILON = 1E-30


def compute_hazard_maps(curves, imls, poes):
    """
    Given a set of hazard curve poes, interpolate a hazard map at the specified
    ``poe``.

    :param curves:
        2D array of floats. Each row represents a curve, where the values
        in the row are the PoEs (Probabilities of Exceedance) corresponding to
        ``imls``. Each curve corresponds to a geographical location.
    :param imls:
        Intensity Measure Levels associated with these hazard ``curves``. Type
        should be an array-like of floats.
    :param poes:
        Value(s) on which to interpolate a hazard map from the input
        ``curves``. Can be an array-like or scalar value (for a single PoE).
    :returns:
        An array of shape N x P, where N is the number of curves and P the
        number of poes.
    """
    poes = numpy.array(poes)

    if len(poes.shape) == 0:
        # `poes` was passed in as a scalar;
        # convert it to 1D array of 1 element
        poes = poes.reshape(1)

    if len(curves.shape) == 1:
        # `curves` was passed as 1 dimensional array, there is a single site
        curves = curves.reshape((1,) + curves.shape)  # 1 x L

    L = curves.shape[1]  # number of levels
    if L != len(imls):
        raise ValueError('The curves have %d levels, %d were passed' %
                         (L, len(imls)))
    result = []
    imls = numpy.log(numpy.array(imls[::-1]))
    for curve in curves:
        # the hazard curve, having replaced the too small poes with EPSILON
        curve_cutoff = [max(poe, EPSILON) for poe in curve[::-1]]
        hmap_val = []
        for poe in poes:
            # special case when the interpolation poe is bigger than the
            # maximum, i.e the iml must be smaller than the minumum
            if poe > curve_cutoff[-1]:  # the greatest poes in the curve
                # extrapolate the iml to zero as per
                # https://bugs.launchpad.net/oq-engine/+bug/1292093
                # a consequence is that if all poes are zero any poe > 0
                # is big and the hmap goes automatically to zero
                hmap_val.append(0)
            else:
                # exp-log interpolation, to reduce numerical errors
                # see https://bugs.launchpad.net/oq-engine/+bug/1252770
                val = numpy.exp(
                    numpy.interp(
                        numpy.log(poe), numpy.log(curve_cutoff), imls))
                hmap_val.append(val)

        result.append(hmap_val)
    return numpy.array(result)


# #########################  GMF->curves #################################### #

# NB (MS): the approach used here will not work for non-poissonian models
def _gmvs_to_haz_curve(gmvs, imls, invest_time, duration):
    """
    Given a set of ground motion values (``gmvs``) and intensity measure levels
    (``imls``), compute hazard curve probabilities of exceedance.

    :param gmvs:
        A list of ground motion values, as floats.
    :param imls:
        A list of intensity measure levels, as floats.
    :param float invest_time:
        Investigation time, in years. It is with this time span that we compute
        probabilities of exceedance.

        Another way to put it is the following. When computing a hazard curve,
        we want to answer the question: What is the probability of ground
        motion meeting or exceeding the specified levels (``imls``) in a given
        time span (``invest_time``).
    :param float duration:
        Time window during which GMFs occur. Another was to say it is, the
        period of time over which we simulate ground motion occurrences.

        NOTE: Duration is computed as the calculation investigation time
        multiplied by the number of stochastic event sets.

    :returns:
        Numpy array of PoEs (probabilities of exceedance).
    """
    # convert to numpy array and redimension so that it can be broadcast with
    # the gmvs for computing PoE values; there is a gmv for each rupture
    # here is an example: imls = [0.03, 0.04, 0.05], gmvs=[0.04750576]
    # => num_exceeding = [1, 1, 0] coming from 0.04750576 > [0.03, 0.04, 0.05]
    imls = numpy.array(imls).reshape((len(imls), 1))
    num_exceeding = numpy.sum(numpy.array(gmvs) >= imls, axis=1)
    poes = 1 - numpy.exp(- (invest_time / duration) * num_exceeding)
    return poes


# ################## utilities for classical calculators ################ #

def get_imts_periods(imtls):
    """
    Returns a list of IMT strings and a list of periods. There is an element
    for each IMT of type Spectral Acceleration, including PGA which is
    considered an alias for SA(0.0). The lists are sorted by period.

    :param imtls: a set of intensity measure type strings
    :returns: a list of IMT strings and a list of periods
    """
    def getperiod(imt):
        return imt[1] or 0
    imts = sorted((from_string(imt) for imt in imtls
                   if imt.startswith('SA') or imt == 'PGA'), key=getperiod)
    return [str(imt) for imt in imts], [imt[1] or 0.0 for imt in imts]


def make_hmap(pmap, imtls, poes):
    """
    Compute the hazard maps associated to the passed probability map.

    :param pmap: hazard curves in the form of a ProbabilityMap
    :param imtls: I intensity measure types and levels
    :param poes: P PoEs where to compute the maps
    :returns: a ProbabilityMap with size (N, I * P, 1)
    """
    I, P = len(imtls), len(poes)
    hmap = ProbabilityMap.build(I * P, 1, pmap)
    for i, imt in enumerate(imtls):
        curves = numpy.array([pmap[sid].array[imtls.slicedic[imt], 0]
                              for sid in pmap.sids])
        data = compute_hazard_maps(curves, imtls[imt], poes)  # array N x P
        for sid, value in zip(pmap.sids, data):
            array = hmap[sid].array
            for j, val in enumerate(value):
                array[i * P + j] = val
    return hmap


def make_uhs(pmap, imtls, poes, nsites):
    """
    Make Uniform Hazard Spectra curves for each location.

    It is assumed that the `lons` and `lats` for each of the `maps` are
    uniform.

    :param pmap:
        a probability map of hazard curves
    :param imtls:
        a dictionary of intensity measure types and levels
    :param poes:
        a sequence of PoEs for the underlying hazard maps
    :returns:
        an composite array containing nsites uniform hazard maps
    """
    P = len(poes)
    array = make_hmap(pmap, imtls, poes).array  # size (N, I x P, 1)
    imts, _ = get_imts_periods(imtls)
    imts_dt = numpy.dtype([(str(imt), F64) for imt in imts])
    uhs_dt = numpy.dtype([(str(poe), imts_dt) for poe in poes])
    uhs = numpy.zeros(nsites, uhs_dt)
    for j, poe in enumerate(map(str, poes)):
        for i, imt in enumerate(imts):
            uhs[poe][imt] = array[:, i * P + j, 0]
    return uhs


def get_gmfs(dstore):
    """
    :param dstore: a datastore
    :returns: a dictionary grp_id, gsid -> gmfa
    """
    oq = dstore['oqparam']
    if 'gmfs' in oq.inputs:  # from file
        logging.info('Reading gmfs from file')
        sitecol, etags, gmfs_by_imt = readinput.get_gmfs(oq)

        # reduce the gmfs matrices to the filtered sites
        for imt in oq.imtls:
            gmfs_by_imt[imt] = gmfs_by_imt[imt][sitecol.indices]

        logging.info('Preparing the risk input')
        return etags, {(0, 'FromFile'): gmfs_by_imt}

    # else from datastore
    rlzs_assoc = dstore['csm_info'].get_rlzs_assoc()
    rlzs = rlzs_assoc.realizations
    sitecol = dstore['sitecol']
    # NB: if the hazard site collection has N sites, the hazard
    # filtered site collection for the nonzero GMFs has N' <= N sites
    # whereas the risk site collection associated to the assets
    # has N'' <= N' sites
    if dstore.parent:
        haz_sitecol = dstore.parent['sitecol']  # N' values
    else:
        haz_sitecol = sitecol
    risk_indices = set(sitecol.indices)  # N'' values
    N = len(haz_sitecol.complete)
    imt_dt = numpy.dtype([(str(imt), F32) for imt in oq.imtls])
    E = oq.number_of_ground_motion_fields
    # build a matrix N x E for each GSIM realization
    gmfs = {(grp_id, gsim): numpy.zeros((N, E), imt_dt)
            for grp_id, gsim in rlzs_assoc}
    for i, rlz in enumerate(rlzs):
        data = group_array(dstore['gmf_data/%04d' % i], 'sid')
        for sid, array in data.items():
            if sid in risk_indices:
                for imti, imt in enumerate(oq.imtls):
                    a = get_array(array, imti=imti)
                    gs = str(rlz.gsim_rlz)
                    gmfs[0, gs][imt][sid, a['eid']] = a['gmv']
    etags = numpy.array(
        sorted([b'scenario-%010d~ses=1' % i
                for i in range(oq.number_of_ground_motion_fields)]))
    return etags, gmfs


def fix_minimum_intensity(min_iml, imts):
    """
    :param min_iml: a dictionary, possibly with a 'default' key
    :param imts: an ordered list of IMTs
    :returns: a numpy array of intensities, one per IMT

    Make sure the dictionary minimum_intensity (provided by the user in the
    job.ini file) is filled for all intensity measure types and has no key
    named 'default'. Here is how it works:

    >>> min_iml = {'PGA': 0.1, 'default': 0.05}
    >>> fix_minimum_intensity(min_iml, ['PGA', 'PGV'])
    array([ 0.1 ,  0.05], dtype=float32)
    >>> sorted(min_iml.items())
    [('PGA', 0.1), ('PGV', 0.05)]
    """
    if min_iml:
        for imt in imts:
            try:
                min_iml[imt] = oqvalidation.getdefault(min_iml, imt)
            except KeyError:
                raise ValueError(
                    'The parameter `minimum_intensity` in the job.ini '
                    'file is missing the IMT %r' % imt)
    if 'default' in min_iml:
        del min_iml['default']
    return F32([min_iml.get(imt, 0) for imt in imts])


gmv_dt = numpy.dtype([('sid', U16), ('eid', U32), ('imti', U8), ('gmv', F32)])


def check_overflow(calc):
    """
    :param calc: an event based calculator

    Raise a ValueError if the number of sites is larger than 65,536 or the
    number of IMTs is larger than 256 or the number of ruptures is larger
    than 4,294,967,296. The limits are due to the numpy dtype used to
    store the GMFs (gmv_dt). They could be relaxed in the future.
    """
    max_ = dict(sites=2**16, events=2**32, imts=2**8)
    num_ = dict(sites=len(calc.sitecol),
                events=len(calc.datastore['events']),
                imts=len(calc.oqparam.imtls))
    for var in max_:
        if num_[var] > max_[var]:
            raise ValueError(
                'The event based calculator is restricted to '
                '%d %s, got %d' % (max_[var], var, num_[var]))
