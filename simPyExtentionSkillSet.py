import random

import simpy

from simpy.resources import resource
from simpy.core import BoundClass, Environment, SimTime

import copy
from commonParams import *
from enum import Enum

counter = 0
time_str = '05/17/2021 00:00:00'


class WorkerStatus(Enum):
    OnBreak = 1
    OnJob = 2
    Idle = 3


class MyEvent():
    def __init__(self):
        self.start = None
        self.end = None
        self.id = None

class BreakEvent(MyEvent):
    def __init__(self, worker_id: int):
        super(BreakEvent, self).__init__()
        self.start = WorkerStatus.OnBreak
        self.end = WorkerStatus.Idle
        self.id = worker_id
    def __str__(self):
        return f'break event for worker {self.id}'


class NewJobEvent(MyEvent):
    def __init__(self, worker_id = None, job_id = None):
        super(NewJobEvent, self).__init__()
        self.start = WorkerStatus.OnJob
        self.end = WorkerStatus.Idle
        self.worker_id = worker_id
        self.job_id = job_id
    def __str__(self):
        return f'job {self.job_id}'


class BatchJobEvent(MyEvent):
    def __init__(self, worker_id = None, jobArray  = None):
        super(BatchJobEvent, self).__init__()
        self.start = WorkerStatus.OnJob
        self.end = WorkerStatus.Idle
        self.worker_id = worker_id
        # contain all the events objects
        self.jobArray = jobArray
        self.nbJob = len(jobArray)
    def __str__(self):
        return f'job {self.jobArray}'

    def jobs(self):
        return self.jobArray

# now lets use call back to change status
class Worker:
    def __init__(self, id = 0, start = 0, end = 0, name = ''):
        self.start = 0
        self.end = end
        self.free = 0
        self.isFree = True
        self.status = WorkerStatus.Idle
        self.id = id
        self.job_id = None
        self.name = name

    def get_status(self):
        return self.status

    def is_free(self, time, duration):
        return time + duration < self.end

    def add_job(self, duration):
        self.free = self.free + duration
        self.isFree = False

    def change_status(self, value: WorkerStatus):
        self.status = value

    def change_status_by_event_start(self, request: 'MyPreemptimeRequest'):
        event = request.event
        print(event.start)
        self.status = event.start

    def change_status_by_event_end(self, request: 'MyPreemptimeRequest'):
        event = request.event
        self.status = event.end

    def __str__(self):
        f = f'worker {self.id} {self.name} status {self.status} on job {self.job_id}'
        return f

    def set_job(self, id):
        self.job_id = id

# the event object contains worker information
class MyPreemptimeRequest(resource.PriorityRequest):
    def __init__(
            self, resource: 'MyPriorityResource', event: MyEvent = None, priority: int = 0, preempt: bool = True
    ):
        global counter
        self.id = copy.deepcopy(counter)
        counter = counter + 1
        self.event = event
        super().__init__(resource, priority, preempt=True)
# i can also reqeust a specific resource
    def __str__(self):
        if (isinstance(self.event, NewJobEvent)):
            s = f'job {self.event.job_id} is waiting'
            return s
        return ''


class WorkerCollection:
    def __init__(self, stage, start = 0, finish = 0):
        self.stage = stage
        self.collection = []
        self.start = start
        self.finish = finish

    def add(self, worker: Worker):
        self.collection.append(worker)
    def __getitem__(self, idx):
        return self.collection[idx]
    def len(self):
        return len(self.collection)

    def get_free_worker(self):
        if False:
            freeWorkers = []
            for worker in self.collection:
                if worker.status == WorkerStatus.Idle:
                    freeWorkers.append(worker)
            if len(freeWorkers) > 0:
                w = random.choices(freeWorkers, [1.0/(len(freeWorkers) * 1.0) for i in range(len(freeWorkers))], k=1)[0]
                return w

            return None

        for worker in self.collection:
            if worker.status == WorkerStatus.Idle:
                return worker
        return None
    def __str__(self):
        return f'stage is {self.stage}'

    def get_shift_start(self):
        return self.start

    def get_duration_in_hours(self):
        return self.finish - self.start


# this is a batch request containing multiple items

class BatchPriorityRequest(resource.PriorityRequest):
    def __init__(
        self, resource: 'BatchPriorityResource',  event: MyEvent = None, priority: int = 0, preempt: bool = True
    ):
        # event is either a job or break event
        global counter
        self.id = copy.deepcopy(counter)
        counter = counter + 1
        self.event = event
        self.requests = []
        super().__init__(resource, priority, preempt = False)

    def __str__(self):
        if isinstance(self.event, BatchJobEvent):
            s = f'priority is {self.priority} for job {self.event.__str__()} reqeust for resource {self.resource}'
        else:
            s = f'priority is {self.priority}'
        return  s
    def update_key(self):
        self.key = (self.priority, self.time, not self.preempt)



class MyPriorityRequest(resource.PriorityRequest):
    def __init__(
        self, resource: 'MyPriorityResource',  event: MyEvent = None, priority: int = 0, preempt: bool = True
    ):
        global counter
        self.id = copy.deepcopy(counter)
        counter = counter + 1
        self.event = event
        super().__init__(resource, priority, preempt = False)
    def __str__(self):
        if isinstance(self.event, NewJobEvent):
            s = f'priority is {self.priority} for job {self.event.__str__()} reqeust for resource {self.resource}'
        else:
            s = f'priority is {self.priority}'
        return  s
    def update_key(self):
        self.key = (self.priority, self.time, not self.preempt)



class BatchRequestRelease(resource.Release):
    def __init__(self, resource: 'Resource', request: BatchPriorityRequest):
        self.worker = request.value
        super().__init__(resource, request)

# relese will call put again for more jobs
class MyRelease(resource.Release):
    def __init__(self, resource: 'Resource', request: MyPriorityRequest):
        self.worker = request.value
        super().__init__(resource, request)


class MyPreemptiveResouce(resource.PreemptiveResource):
    def __init__(self, env: simpy.Environment, workers: WorkerCollection):
        self.workers = workers
        capacity = workers.len()
        # when this is true, we will preempt a resource even list is not full if we must use this resource
        self.strickUserChoice = True
        super().__init__(env, capacity)


    def _do_put(  # type: ignore[override] # noqa: F821
        self, event: MyPreemptimeRequest
    ) -> None:
        # if I want a specifc worker, even capcity is not full, I can still preemp
        # how do I know which requet to preemp?
        newRequestEvent = event.event
        newRequestedWorker = None
        preempNotFull = False
        # this can be used as a prefereed user
        if (self.strickUserChoice):
            if newRequestEvent and isinstance(newRequestEvent, NewJobEvent):
                if newRequestEvent.worker_id is not None:
                    # a request is a job as a user
                    id = newRequestEvent.worker_id
                    newRequestedWorker = self.workers[id]
                for preempt in self.users:
                    requestEvent = preempt.event
                    if requestEvent and isinstance(requestEvent, NewJobEvent):
                        currentWorker = preempt._value
                        if newRequestedWorker == currentWorker:
                            if preempt.key > event.key:
                                # this user is busy, but we can inrrupt
                                currentWorker.change_status(requestEvent.end)
                                self.users.remove(preempt)
                                preempt.proc.interrupt(  # type: ignore
                                        resource.Preempted(
                                            by=event.proc,
                                            usage_since=preempt.usage_since,
                                            resource=self,
                                        )
                                    )
                                preempNotFull = True
                                break
        # a normal preempt, onl preempt when  resoures are full
        if not preempNotFull and (len(self.users) >= self.capacity and event.preempt):
            # Check if we can preempt another process
            preempt = sorted(self.users, key=lambda e: e.key)[-1]
            if preempt.key > event.key:
                # need to free the preep user
                print(f'preemp request id is {preempt.id}')
                requestEvent = preempt.event
                if requestEvent and isinstance(requestEvent, BreakEvent):
                    # preemp is a break event, time to change the status here?
                    worker_id = requestEvent.id
                    self.workers[worker_id].change_status(requestEvent.end)
                if requestEvent and isinstance(requestEvent, NewJobEvent):
                    # preemp is a break event, time to change the status here?
                    worker = preempt._value
                    worker.change_status(requestEvent.end)



                self.users.remove(preempt)
                preempt.proc.interrupt(  # type: ignore
                    resource.Preempted(
                        by=event.proc,
                        usage_since=preempt.usage_since,
                        resource=self,
                    )
                )
        # the reqeut contains the event type object
        # event here is a request
        requestEvent = event.event
        preassigned_worker = None
        if requestEvent and isinstance(requestEvent, NewJobEvent):
            if requestEvent.worker_id is not None:
                # I request a specific worker
                id = requestEvent.worker_id
                worker = self.workers[id]

                if worker.status != WorkerStatus.Idle:
                    # this will proceed to next request on the put_queue see if we can satisify next request
                    # if I dont have to use this worker, just find the next free worker
                    # this means this request will wait until this user become free, otherwise it will be on the put_queue forever
                    if self.strickUserChoice:
                        return True
                else:
                    preassigned_worker = worker

        #assert (preassigned_worker == None)
        if len(self.users) < self.capacity:
            self.users.append(event)
            event.usage_since = self._env.now
            # request should know which worker has the job

            if requestEvent and isinstance(requestEvent, BreakEvent):
                # meet this break request, change status to on break
                worker_id = requestEvent.id
                self.workers[worker_id].change_status(requestEvent.start)
                event.succeed()
            if requestEvent and isinstance(requestEvent, NewJobEvent):
                # if it is a new job event, need to assign the correct worker
                # assign the worker to the request succeed
                if preassigned_worker:
                    w = preassigned_worker

                else:
                    w = self.workers.get_free_worker()
                requestEvent.id = w.id
                event.succeed(w)
                # change status to onjob
                w.change_status(requestEvent.start)

    # odd is here
    def _do_get(self, event: MyRelease) -> None:
        try:
            # change worer state
            # the release should hold the infomration of the worker
            # my release knows the request
            request = event.request
            requestEvent = request.event
            # break is finished
            if (event.request in self.users):
                # at this point, any request will know which worker took the request
                worker_id = requestEvent.id
                self.workers[worker_id].change_status(requestEvent.end)

                #if requestEvent and isinstance(requestEvent, BreakEvent):
                #    worker_id = requestEvent.id
                    # change status to idle
                #    self.workers[worker_id].change_status(requestEvent.end)
                #if requestEvent and isinstance(requestEvent, NewJobEvent):
                    # here the request event also know the worker
                    #worker = request._value
                #    worker = workers[requestEvent.id]
                    # change status to idle
                #    worker.change_status(requestEvent.end)
            self.users.remove(event.request)  # type: ignore
        except ValueError:
            pass
        # relse event know which worker to release
        event.succeed()

    request = BoundClass(MyPreemptimeRequest)
    release = BoundClass(MyRelease)

class MyPriorityResource(simpy.PriorityResource):

    def __init__(self, env: simpy.Environment, workers: WorkerCollection, tag = ''):
        self.workers = workers
        self.tag = tag
        capacity = workers.len()
        super().__init__(env, capacity)

    def __str__(self):
        return f'resource {self.workers.stage}'

    def get_workers(self) -> WorkerCollection:
        return self.workers

    def at_least_one_free(self):
        return self.workers.get_free_worker() is not None

    def _do_put(self, event: MyPriorityRequest) -> None:
        # no preemption, just get most important request
        # we cant append multiple break events for the same resource
        requestEvent = event.event
        if requestEvent and isinstance(requestEvent, BreakEvent):
            worker_id = requestEvent.id
            if self.workers[worker_id].get_status() != WorkerStatus.Idle:
                # for break event ,we want to continue look for other break events
                # this break event can come for a worker is busy, then we try other break event for other worker
                # if i dont do this, say I want break event for worker 5, but worker 1,2,3,4 also request break but they are busy
                # worker 5 break event still never be called back, it will be on the queue
                # as worker 1,2,3, 4 finish, it will call this, to execute the first break on the queue then just return

                return True

        if len(self.users) < self.capacity:
            self.users.append(event)
            event.usage_since = self._env.now
            requestEvent = event.event
            if requestEvent and isinstance(requestEvent, BreakEvent):
                # meet this break request, change status to on break
                worker_id = requestEvent.id
                self.workers[worker_id].change_status(requestEvent.start)
                event.succeed()  # break event dont need return value
            if requestEvent and isinstance(requestEvent, NewJobEvent):
                # step 1, just get a free worker if possible
                # if we dont care about which worker,,just pick one to mark it for this request
                # now this job request is linked to that worker, any new job reqeust added to this user list
                # will not use that worker until its free
                w = self.workers.get_free_worker()
                requestEvent.id = w.id
                w.set_job(requestEvent.job_id)
                event.succeed(w)
                # change status to onjob
                w.change_status(requestEvent.start)

    def _do_get(self, event: MyRelease) -> None:
        try:
            # change worer state
            # the release should hold the infomration of the worker
            # my release knows the request
            request = event.request
            requestEvent = request.event
            # break is finished
            if (event.request in self.users):
                # at this point, any request will know which worker took the request
                worker_id = requestEvent.id
                w: Worker = self.workers[worker_id]
                #print(f'free {self.workers} for worker {worker_id}')
                w.change_status(requestEvent.end)
                w.set_job(None)

                #if requestEvent and isinstance(requestEvent, BreakEvent):
                #    worker_id = requestEvent.id
                    # change status to idle
                #    self.workers[worker_id].change_status(requestEvent.end)
                #if requestEvent and isinstance(requestEvent, NewJobEvent):
                    # here the request event also know the worker
                    #worker = request._value
                #    worker = workers[requestEvent.id]
                    # change status to idle
                #    worker.change_status(requestEvent.end)
            self.users.remove(event.request)  # type: ignore
        except ValueError:
            pass
        # relse event know which worker to release
        event.succeed()

    request = BoundClass(MyPriorityRequest)
    release = BoundClass(MyRelease)

class BatchPriorityResource(simpy.PriorityResource):

    def __init__(self, env: simpy.Environment, workers: WorkerCollection):
        self.workers = workers
        capacity = workers.len()
        super().__init__(env, capacity)

    def __str__(self):
        return f'resource {self.workers.stage}'

    def get_workers(self) -> WorkerCollection:
        return self.workers

    def at_least_one_free(self):
        return self.workers.get_free_worker() is not None

    def _do_put(self, event: BatchPriorityRequest) -> None:
        # no preemption, just get most important request

        # we cant append multiple break events for the same resource
        requestEvent = event.event
        if requestEvent and isinstance(requestEvent, BreakEvent):

            #assert(len(self.users)  == 0)
            if len(self.users)  >  0:
                # for break event ,we want to continue look for other break events
                # this break event can come for a worker is busy, then we try other break event for other worker
                # if i dont do this, say I want break event for worker 5, but worker 1,2,3,4 also request break but they are busy
                # worker 5 break event still never be called back, it will be on the queue
                # as worker 1,2,3, 4 finish, it will call this, to execute the first break on the queue then just return

                return True
        debug_output(f'request batch res for {requestEvent} at {format_time(self._env.now)}')
        if isinstance(requestEvent, BatchJobEvent):
            spaceLeft = self.capacity - len(self.users)
            jobs = requestEvent.jobs()
            nbJobs = len(jobs)
            # enough space for this case batch job
            if nbJobs <= spaceLeft:
                assignedWorkersId = []
                assignedWorkers = []
                for i in range(nbJobs):
                    self.users.append(event)
                    w = self.workers.get_free_worker()
                    assignedWorkersId.append(w.id)
                    assignedWorkers.append(w)
                    w.set_job(jobs[i])

                    # change status to onjob
                    w.change_status(requestEvent.start)
                requestEvent.id = assignedWorkersId
                event.succeed(assignedWorkers)

            # when a reqeust for a job is made, it will go through the first reqeust in the queue
            # if the first reqeust cant be met, dont even bother to go to the next one for batching
            return None
        assert(isinstance(requestEvent, BreakEvent))
        assert len(self.users)  == 0, f'{format_time(self._env.now)}'
        if len(self.users) == 0:
            for i in range(self.capacity):
                self.users.append(event)
            event.usage_since = self._env.now
            requestEvent = event.event
            for idx in range(self.workers.len()):
                self.workers[idx].change_status(requestEvent.start)
            event.succeed()  # break event dont need return value
            return None
        else:
            return None

    def _do_get(self, event: MyRelease) -> None:
        try:
            debug_output(f'releast at {format_time(self._env.now)}')
            # change worer state
            # the release should hold the infomration of the worker
            # my release knows the request
            request = event.request
            requestEvent = request.event
            # break is finished
            if isinstance(requestEvent, BatchJobEvent):
                worker_ids = requestEvent.id
                for worker_id in worker_ids:
                    self.users.remove(event.request)  # type: ignore
                    w: Worker = self.workers[worker_id]
                    # print(f'free {self.workers} for worker {worker_id}')
                    w.change_status(requestEvent.end)
                    w.set_job(None)
                event.succeed()
                assert(len(self.get_queue) == 1)
                return None
            assert(isinstance(requestEvent, BreakEvent))
            if (event.request in self.users):
                # at this point, any request will know which worker took the request

                for idx in range(self.workers.len()):
                    w: Worker = self.workers[idx]
                    #print(f'free {self.workers} for worker {worker_id}')
                    w.change_status(requestEvent.end)
                    w.set_job(None)
                    self.users.remove(event.request)  # type: ignore
                event.succeed()
                return
        except ValueError:
            pass
        # relse event know which worker to release
        event.succeed()

    request = BoundClass(BatchPriorityRequest)
    release = BoundClass(BatchRequestRelease)


class NewBatchPriorityResource(simpy.PriorityResource):
    def __init__(self, env: simpy.Environment, workers: WorkerCollection):
        self.workers = workers
        capacity = workers.len()
        super().__init__(env, capacity)

    def get_workers(self) -> WorkerCollection:
        return self.workers

    def _do_put(self, event: BatchPriorityRequest) -> None:
        # no preemption, just get most important request

        requestEvent = event.event
        if requestEvent and isinstance(requestEvent, BreakEvent):
            # put a break event for every worker
            # now all jobs should have been finished?
            if len(self.users) > 0:
                debug_output('h')
            if len(self.users) == 0:
                #debug_output(f'put all batch workers to sleep', {format_time(self._env.now)})
                for i in range(self.capacity):
                    self.users.append(event)
                event.usage_since = self._env.now
                for idx in range(self.workers.len()):
                    self.workers[idx].change_status(requestEvent.start)
                event.succeed()  # break event dont need return value

        if isinstance(requestEvent, BatchJobEvent):
            spaceLeft = self.capacity - len(self.users)
            jobs = requestEvent.jobs()
            nbJobs = len(jobs)
            # enough space for this case batch job
            if nbJobs <= spaceLeft:
                assignedWorkersId = []
                assignedWorkers = []
                for i in range(nbJobs):
                    self.users.append(event)
                    w = self.workers.get_free_worker()
                    assignedWorkersId.append(w.id)
                    assignedWorkers.append(w)
                    w.set_job(jobs[i])

                    # change status to onjob
                    w.change_status(requestEvent.start)
                requestEvent.id = assignedWorkersId
                event.usage_since = self._env.now
                event.succeed(assignedWorkers)
            return True
    def _do_get(self, event: MyRelease) -> None:
        try:
            debug_output(f'releast at {format_time(self._env.now)}')
            # change worer state
            # the release should hold the infomration of the worker
            # my release knows the request
            request = event.request
            requestEvent = request.event
            # break is finished
            if isinstance(requestEvent, BatchJobEvent):
                worker_ids = requestEvent.id
                for worker_id in worker_ids:
                    self.users.remove(event.request)  # type: ignore
                    w: Worker = self.workers[worker_id]
                    # print(f'free {self.workers} for worker {worker_id}')
                    w.change_status(requestEvent.end)
                    w.set_job(None)
                event.succeed()
                assert (len(self.get_queue) == 1)
                return None

            assert (isinstance(requestEvent, BreakEvent))
            if (event.request in self.users):
                # at this point, any request will know which worker took the request

                for w in self.workers.collection:
                    w: Worker = w
                    w.change_status(requestEvent.end)
                    w.set_job(None)
                    self.users.remove(event.request)  # type: ignore
                event.succeed()
                return  None
        except ValueError:
            pass
        # relse event know which worker to release
        event.succeed()

        # try next bat
    request = BoundClass(BatchPriorityRequest)
    release = BoundClass(BatchRequestRelease)

def checkDiff(file1, file2):
    import difflib
    print('check diff between ', file1, file2)
    with open(file1) as file_1:
        file_1_text = file_1.readlines()
    with open(file2) as file_2:
        file_2_text = file_2.readlines()
    # Find and print the diff:
    for line in difflib.unified_diff(
            file_1_text, file_2_text, fromfile=file1,
            tofile=file2, lineterm=''):
        print(line)

def debug_output(str, force_show=False):
    show = False
    if show or force_show:
        print(str)





