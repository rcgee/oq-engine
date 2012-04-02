# nhlib: A New Hazard Library
# Copyright (C) 2012 GEM Foundation
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Module :mod:`nhlib.geo.line` defines :class:`Line`.
"""
import numpy

from nhlib.geo import _utils as utils


class Line(object):
    """
    This class represents a geographical line, which is basically
    a sequence of geographical points.

    A line is defined by at least one point. The surface projection
    of a line cannot intersect itself (depth dimension is neglected
    to check if a line intersects itself or not).

    :param points:
        The sequence of points defining this line.
    :type points:
        list of :class:`~nhlib.geo.point.Point` instances
    """

    def __init__(self, points):
        self.points = utils.clean_points(points)

        if len(self.points) < 1:
            raise ValueError("One point needed to create a line!")

        lats = [point.latitude for point in self.points]
        lons = [point.longitude for point in self.points]
        if utils.line_intersects_itself(lons, lats):
            raise ValueError("Line intersects itself!")

    def __eq__(self, other):
        """
        >>> from nhlib.geo.point import Point
        >>> points = [Point(1, 2), Point(3, 4)]; Line(points) == Line(points)
        True
        >>> Line(points) == Line(list(reversed(points)))
        False
        """
        return self.points == other.points

    def __ne__(self, other):
        """
        >>> from nhlib.geo.point import Point
        >>> Line([Point(1, 2)]) != Line([Point(1, 2)])
        False
        >>> Line([Point(1, 2)]) != Line([Point(2, 1)])
        True
        """
        return not self.__eq__(other)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, key):
        return self.points.__getitem__(key)

    def on_surface(self):
        """
        Check if this line is defined on the surface (i.e. all points
        are on the surfance, depth=0.0).

        :returns:
            True if this line is on the surface, false otherwise.
        :rtype:
            boolean
        """
        return all(point.on_surface() for point in self.points)

    def average_azimuth(self):
        """
        Calculate and return weighted average azimuth of all line's segments
        in decimal degrees.

        Uses formula from
        `http://en.wikipedia.org/wiki/Mean_of_circular_quantities`_.

        >>> from nhlib.geo.point import Point as P
        >>> str(Line([P(0, 0), P(1e-5, 1e-5)]).average_azimuth())
        '45.0'
        >>> str(Line([P(0, 0), P(0, 1e-5), P(1e-5, 1e-5)]).average_azimuth())
        '45.0'
        >>> line = Line([P(0, 0), P(-2e-5, 0), P(-2e-5, 1.154e-5)])
        >>> '%.1f' % line.average_azimuth()
        '300.0'
        """
        if len(self.points) == 2:
            return self.points[0].azimuth(self.points[1])
        points = iter(self.points)
        prev_point = next(points)
        azimuths = []
        distances = []
        for point in points:
            # collect all the segments' lengths and azimuths
            assert point.depth == prev_point.depth
            azimuth, _, distance = utils.GEOD.inv(
                prev_point.longitude, prev_point.latitude,
                point.longitude, point.latitude
            )
            azimuths.append(azimuth)
            distances.append(distance)
            prev_point = point
        azimuths, distances = numpy.radians(azimuths), numpy.array(distances)
        # convert polar coordinates to Cartesian ones and calculate
        # the average coordinate of each component
        avg_x = numpy.mean(distances * numpy.sin(azimuths))
        avg_y = numpy.mean(distances * numpy.cos(azimuths))
        # find the mean azimuth from that mean vector
        azimuth = numpy.degrees(numpy.arctan2(avg_x, avg_y))
        if azimuth < 0:
            azimuth += 360
        return azimuth

    def resample(self, section_length):
        """
        Resample this line into sections.

        The first point in the resampled line corresponds
        to the first point in the original line.

        Starting from the first point in the original line, a line
        segment is defined as the line connecting the last point in the
        resampled line and the next point in the original line.
        The line segment is then split into sections of length equal to
        ``section_length``. The resampled line is obtained
        by concatenating all sections.

        The number of sections in a line segment is calculated as follows:
        ``round(segment_length / section_length)``.

        Note that the resulting line has a length that is an exact multiple of
        ``section_length``, therefore its length is in general smaller
        or greater (depending on the rounding) than the length
        of the original line.

        For a straight line, the difference between the resulting length
        and the original length is at maximum half of the ``section_length``.
        For a curved line, the difference my be larger,
        because of corners getting cut.

        :param section_length:
            The length of the section, in km.
        :type section_length:
            float
        :returns:
            A new line resampled into sections based on the given length.
        :rtype:
            An instance of :class:`Line`
        """

        if len(self.points) < 2:
            return Line(self.points)

        resampled_points = []

        # 1. Resample the first section. 2. Loop over the remaining points
        # in the line and resample the remaining sections.
        # 3. Extend the list with the resampled points, except the first one
        # (because it's already contained in the previous set of
        # resampled points).

        resampled_points.extend(self.points[0].equally_spaced_points(
                self.points[1], section_length))

        # Skip the first point, it's already resampled
        for i in range(2, len(self.points)):
            points = resampled_points[-1].equally_spaced_points(
                    self.points[i], section_length)

            resampled_points.extend(points[1:])

        return Line(resampled_points)

    def resample_to_num_points(self, num_points):
        """
        Resample the line to a specified number of points.

        :param num_points:
            Integer number of points the resulting line should have.
        :returns:
            A new line with that many points as requested.

        Calculates the length of the original line, divides it by number
        of segments in the resulting line (which is less than num_points
        by 1) and calls :meth:`resample` with the calculated section length.
        """
        length = 0
        for i, point in enumerate(self.points):
            if i != 0:
                length += point.distance(self.points[i - 1])
        section_length = length / (num_points - 1)
        return self.resample(section_length)
