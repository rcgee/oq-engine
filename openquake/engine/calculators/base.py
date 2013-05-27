# Copyright (c) 2010-2013, GEM Foundation.
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

import math

import kombu

import openquake.engine
from openquake.engine import logs
from openquake.engine.utils import config, tasks, general

# Routing key format string for communication between tasks and the control
# node.
ROUTING_KEY_FMT = 'oq.job.%(job_id)s.tasks'


class Calculator(object):
    """
    Base class for all calculators.

    :param job: :class:`openquake.engine.db.models.OqJob` instance.
    """

    #: The core calculation Celery task function, which accepts the arguments
    #: generated by :func:`task_arg_gen`.
    core_calc_task = None

    def __init__(self, job):
        self.job = job

        self.progress = dict(total=0, computed=0, in_queue=0)

    def task_arg_gen(self, block_size):
        """
        Generator function for creating the arguments for each task.

        Subclasses must implement this.

        :param int block_size:
            The number of work items per task (sources, sites, etc.).
        """
        raise NotImplementedError

    def block_size(self):
        """
        Number of work items per task.

        Subclasses must implement this.
        """
        raise NotImplementedError()

    def concurrent_tasks(self):
        """
        Number of tasks to be in queue at any given time.

        Subclasses must implement this.
        """
        raise NotImplementedError()

    def parallelize(self, task_func, task_arg_gen):
        """
        Given a callable and a task arg generator, apply the callable to
        the arguments in parallel. To save memory the tasks are spawned in
        blocks with maximum size defined by the method .concurrent_tasks().
        It is possible to pass a function side_effect(ret) which takes the
        return value of the callable and does something with it, such as
        saving or printing it. The order is not preserved.

        :param task_func: a `celery` task callable
        :param task_args: an iterable over positional arguments

        NB: if the environment variable OQ_NO_DISTRIBUTE is set the
        tasks are run sequentially in the current process.
        """
        taskname = task_func.__name__
        logs.LOG.progress('building arglist')
        arglist = list(task_arg_gen)
        total = len(arglist)
        logs.LOG.progress('spawning %d tasks of kind %s', total, taskname)
        ntasks = 0
        for argblock in general.block_splitter(
                arglist, self.concurrent_tasks()):
            tasks.parallelize(task_func, argblock, lambda _: None)
            ntasks += len(argblock)
            percent = math.ceil(float(ntasks) / total * 100)
            logs.LOG.progress('> %s %3d%% complete', taskname, percent)

    def get_task_complete_callback(self, task_arg_gen, block_size,
                                   concurrent_tasks):
        """
        Create the callback which responds to a task completion signal. In some
        cases, the response is simply to enqueue the next task (if there is
        any work left to be done).

        :param task_arg_gen:
            The task arg generator, so the callback can get the next set of
            args and enqueue the next task.
        :param int block_size:
            The (maximum) number of work items to pass to a given task.
        :param int concurrent_tasks:
            The (maximum) number of tasks that should be in queue at any time.
        :return:
            A callback function which responds to a task completion signal.
            A response typically includes enqueuing the next task and updating
            progress counters.
        """

        def callback(body, message):
            """
            :param dict body:
                ``body`` is the message sent by the task. The dict should
                contain 2 keys: `job_id` and `num_items` (to indicate the
                number of sources computed).

                Both values are `int`.
            :param message:
                A :class:`kombu.transport.pyamqplib.Message`, which contains
                metadata about the message (including content type, channel,
                etc.). See kombu docs for more details.
            """
            job_id = body['job_id']
            num_items = body['num_items']

            assert job_id == self.job.id
            self.progress['computed'] += num_items

            self.task_completed_hook(body)

            logs.log_percent_complete(job_id, "hazard")
            logs.log_percent_complete(job_id, "risk")

            # Once we receive a completion signal, enqueue the next
            # piece of work (if there's anything left to be done).
            try:
                queue_next(self.core_calc_task, task_arg_gen.next())
            except StopIteration:
                # There are no more tasks to dispatch; now we just need
                # to wait until all tasks signal completion.
                self.progress['in_queue'] -= 1

            message.ack()
            logs.LOG.info('A task was completed. Tasks now in queue: %s'
                          % self.progress['in_queue'])

        return callback

    def task_completed_hook(self, body):
        """
        Performs an action when a task is completed successfully.
        :param dict body: the message sent by the task. It contains at least
        the keys `job_id` and `num_items`. They idea is to add additional
        keys and then process them in the hook. Notice that the message
        is sent by using
        `openquake.engine.calculators.base.signal_task_complete`.
        """
        pass

    def pre_execute(self):
        """
        Override this method in subclasses to record pre-execution stats,
        initialize result records, perform detailed parsing of input data, etc.
        """

    # this method is completely overridden in the event based calculator
    def execute(self):
        """
        Calculation work is parallelized over sources, which means that each
        task will compute hazard for all sites but only with a subset of the
        seismic sources defined in the input model.

        The general workflow is as follows:

        1. Fill the queue with an initial set of tasks. The number of initial
        tasks is configurable using the `concurrent_tasks` parameter in the
        `[hazard]` section of the OpenQuake config file.

        2. Wait for tasks to signal completion (via AMQP message) and enqueue a
        new task each time another completes. Once all of the job work is
        enqueued, we just wait until all of the tasks conclude.
        """
        if openquake.engine.no_distribute():
            logs.LOG.warn('Calculation task distribution is disabled')
        # The following two counters are in a dict so that we can use them in
        # the closures below.
        # When `self.progress['compute']` becomes equal to
        # `self.progress['total']`, `execute` can conclude.

        task_gen = self.task_arg_gen(self.block_size())
        exchange, conn_args = exchange_and_conn_args()

        routing_key = ROUTING_KEY_FMT % dict(job_id=self.job.id)
        task_signal_queue = kombu.Queue(
            'tasks.job.%s' % self.job.id, exchange=exchange,
            routing_key=routing_key, durable=False, auto_delete=True)

        with kombu.BrokerConnection(**conn_args) as conn:
            task_signal_queue(conn.channel()).declare()
            with conn.Consumer(
                task_signal_queue,
                callbacks=[self.get_task_complete_callback(
                    task_gen, self.block_size(), self.concurrent_tasks())]):

                # First: Queue up the initial tasks.
                for _ in xrange(self.concurrent_tasks()):
                    try:
                        queue_next(self.core_calc_task, task_gen.next())
                    except StopIteration:
                        # If we get a `StopIteration` here, that means we have
                        # a number of tasks < concurrent_tasks.
                        # This basically just means that we could be
                        # under-utilizing worker node resources.
                        break
                    else:
                        self.progress['in_queue'] += 1

                logs.LOG.info('Tasks now in queue: %s'
                              % self.progress['in_queue'])

                while (self.progress['computed'] < self.progress['total']):
                    # This blocks until a message is received.
                    # Once we receive a completion signal, enqueue the next
                    # piece of work (if there's anything left to be done).
                    # (The `task_complete_callback` will handle additional
                    # queuing.)
                    conn.drain_events()
        logs.LOG.progress("calculation 100% complete")

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

    def export(self, *args, **kwargs):
        """Implement this method in subclasses to write results
           to places other than the database."""

    def clean_up(self, *args, **kwargs):
        """Implement this method in subclasses to perform clean-up actions
           like garbage collection, etc."""


def exchange_and_conn_args():
    """
    Helper method to setup an exchange for task communication and the args
    needed to create a broker connection.
    """

    amqp_cfg = config.get_section('amqp')
    exchange = kombu.Exchange(amqp_cfg['task_exchange'], type='direct')

    conn_args = {
        'hostname': amqp_cfg['host'],
        'userid': amqp_cfg['user'],
        'password': amqp_cfg['password'],
        'virtual_host': amqp_cfg['vhost'],
    }

    return exchange, conn_args


def queue_next(task_func, task_args):
    """
    :param task_func:
        A Celery task function, to be enqueued with the next set of args in
        ``task_arg_gen``.
    :param task_args:
        A set of arguments which match the specified ``task_func``.

    .. note::
        This utility function was added to make for easier mocking and testing
        of the "plumbing" which handles task queuing (such as the various "task
        complete" callback functions).
    """
    if openquake.engine.no_distribute():
        task_func(*task_args)
    else:
        task_func.apply_async(task_args)


def signal_task_complete(**kwargs):
    """
    Send a signal back through a dedicated queue to the 'control node' to
    notify of task completion and the number of items processed.

    Signalling back this metric is needed to tell the control node when it can
    conclude its `execute` phase.

    :param kwargs:
        Arbitrary message parameters. Anything in this dict will go into the
        "task complete" message.

        Typical message parameters would include `job_id` and `num_items` (to
        indicate the number of work items that the task has processed).

        .. note::
            `job_id` is required for routing the message. All other parameters
            can be treated as optional.
    """
    msg = kwargs
    # here we make the assumption that the job_id is in the message kwargs
    job_id = kwargs['job_id']

    exchange, conn_args = exchange_and_conn_args()

    routing_key = ROUTING_KEY_FMT % dict(job_id=job_id)

    with kombu.BrokerConnection(**conn_args) as conn, conn.Producer(
            exchange=exchange, routing_key=routing_key,
            serializer='pickle') as producer:
        producer.publish(msg)
