# Copyright (c) 2010-2014, GEM Foundation.
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

"""Base code for calculator classes."""

from openquake.engine import logs
from openquake.engine.export import core
from openquake.engine.performance import EnginePerformanceMonitor
from openquake.engine.utils import config

from openquake.commonlib.source import TrtModel


class Calculator(object):
    """
    Base class for all calculators.

    :param job: :class:`openquake.engine.db.models.OqJob` instance.
    """

    #: The core calculation Celery task function, which accepts the arguments
    #: generated by :func:`task_arg_gen`.
    core_calc_task = None

    def __init__(self, job, monitor=None):
        self.job = job
        self.monitor = monitor or EnginePerformanceMonitor(
            '', job.id, flush=True)
        self.num_tasks = None
        self._task_args = []
        # parameters from openquake.cfg
        self.concurrent_tasks = int(
            config.get('celery', 'concurrent_tasks'))
        self.max_input_weight = float(
            config.get('hazard', 'max_input_weight'))
        self.max_output_weight = float(
            config.get('hazard', 'max_output_weight'))
        TrtModel.POINT_SOURCE_WEIGHT = float(
            config.get('hazard', 'point_source_weight'))

    def task_arg_gen(self):
        """
        Generator function for creating the arguments for each task.

        Subclasses must implement this.
        """
        raise NotImplementedError

    def pre_execute(self):
        """
        Override this method in subclasses to record pre-execution stats,
        initialize result records, perform detailed parsing of input data, etc.
        """

    def execute(self):
        """
        Run the core_calc_task in parallel
        """
        raise NotImplementedError

    def post_execute(self):
        """
        Override this method in subclasses to any necessary post-execution
        actions, such as the consolidation of partial results.
        """

    def post_process(self):
        """
        Override this method in subclasses to perform post processing steps,
        such as computing mean results from a set of curves or plotting maps.
        """

    def _get_outputs_for_export(self):
        """
        Util function for getting :class:`openquake.engine.db.models.Output`
        objects to be exported.

        Gathers all outputs for the job, but filters out `hazard_curve_multi`
        outputs if this option was turned off in the calculation profile.
        """
        outputs = core.get_outputs(self.job.id)
        if not getattr(self.oqparam, 'export_multi_curves', None):
            outputs = outputs.exclude(output_type='hazard_curve_multi')
        return outputs

    def export(self, *args, **kwargs):
        """
        If requested by the user, automatically export all result artifacts to
        the specified format. (NOTE: The only export format supported at the
        moment is NRML XML.

        :param exports:
            Keyword arg. List of export types.
        :returns:
            A list of the export filenames, including the absolute path to each
            file.
        """
        exported_files = []

        with logs.tracing('exports'):
            export_dir = self.job.get_param('export_dir')
            export_type = kwargs['exports']
            if export_type:
                outputs = self._get_outputs_for_export()
                for output in outputs:
                    with self.monitor('exporting %s to %s'
                                      % (output.output_type, export_type)):
                        fname = core.export(output.id, export_dir, export_type)
                        if fname:
                            logs.LOG.info('exported %s', fname)
                            exported_files.append(fname)

        return exported_files

    def clean_up(self, *args, **kwargs):
        """Implement this method in subclasses to perform clean-up actions
           like garbage collection, etc."""
