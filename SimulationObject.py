import ast
import collections
import statistics


from simPyExtentionSkillSet import *

import time

from commonParams import *
singleUseTag = 'single'
multiUseTag = 'multiUse'
noResNeeded =  'No resource needed'
onLocal = True
fromdBRegression = False

resource_type = collections.namedtuple('resource_type', 'transition capacity resource')
time_bin_type = collections.namedtuple('time_bin', 'start finish')
team_type = collections.namedtuple('team_type', 'teamMember events')
batch_type = collections.namedtuple('batch', 'start finish duration resource')
timeWindow_type =  collections.namedtuple('windows', 'timeIndex start end')
dayTimeWindow_type = collections.namedtuple('windows', 'day timeIndex start end')
staff_rostering = collections.namedtuple('staff', 'timeIndex resource worker_id')
local_search_value = collections.namedtuple('obj','average, resource')
product_count_type = collections.namedtuple('productCount', 'day time resource')



def strip_suffix(name = 'process_0_1'):
    try:
        index = name.index('_')
        if index is not None:
            stripped_name = name[0:index]
            return stripped_name
    except ValueError:
        return name
strip_suffix()
def import_csv(file_path, checkLog=False):
    print(f'start loading {file_path}')

    eventlog = pandas.read_csv(file_path, sep=',', skipinitialspace=True)
    if checkLog:
        num_cases = len(eventlog.case_id.unique().tolist())
        num_events = len(eventlog.event_id.to_list())
        print("Number of events: {}\nNumber of cases: {}".format(num_events, num_cases))

        print(list(eventlog))
        uniqueEvents = eventlog.event_id.unique().tolist()
        print(f'unique events are {len(uniqueEvents)}')
        print('info')
        print(eventlog.duplicated())
    return eventlog

def bin_events(log):
    log.formated_start_time = pandas.to_datetime(log.formated_start_time)

    day_data_windows = {}  # index by day, save array of data windows
    dayList = pandas.date_range(start="2021-05-17", end="2022-06-02").to_pydatetime().tolist()

    for day in dayList:

        month = day.month
        if month < 10:
            month = '0' + str(month)
        datstr = day.day
        if datstr < 10:
            datstr = '0' + str(datstr)
        date_start_loading = f'{day.year}-{month}-{datstr}'
        date_finish_loading = f'{day.year}-{month}-{datstr}'
        time_start_loading = ' '.join([date_start_loading, '00:00:00'])
        time_end_loading = ' '.join([date_finish_loading, '23:59:59'])

        day_data_windows[date_start_loading] = time_bin_type(start=time_start_loading, finish=time_end_loading)

        # group events by data bin
    case_arrival_times = dict(log.groupby(by='case_id').formated_start_time.min())
    job_bin = {}
    for idx, data_window in day_data_windows.items():
        log['arrival_time'] = log['case_id']
        log['arrival_time'] = log["arrival_time"].map(case_arrival_times)
        temp = log[(log.arrival_time >= data_window.start) & (log.arrival_time < data_window.finish)]
        if temp.empty: continue
        job_bin[idx] = temp

    return job_bin


class UserCount:
    def __init__(self, key: product_count_type,group: str):
        self.key = key
        self.group = group
        self.count = 0

    def accumulate(self):
        self.count += 1

class UtilisationCount:
    def __init__(self, group: str):
        self.group = group
        self.value = 0

    def accumulate(self, value):
        self.value += value

class Event():
    def __init__(self, id = 0, start = 0, finish = 0, parents = [], name = '', case_id = 0, is_start = False,  isDone= False):
        # actual is the event with id
        # name is the event name without id, representing the group, for exmaple, grossing_0 -> grossing (group)
        self.id = id
        self.start = start
        self.finish = finish
        self.parents = parents
        self.duration = finish - start
        #self.duration = 35
        self.actualName = name

        self.group = strip_suffix(name)

        self.case_id = case_id
        self.is_start = is_start
        self.simulated_start = 0
        self.simulated_end = 0
        self.useResource = None
        self.resourceTimeIndex = None

        self.ready_time = None
        self.isDone = isDone

    def set_resource(self, r):
        self.useResource = r

    def get_resource(self):
        return self.useResource

    def set_resource_time_index(self, idx):
        self.resourceTimeIndex = idx

    def get_resource_time_index(self):
        return self.resourceTimeIndex

    def __str__(self):
        return f'{self.case_id}, activity = {self.actualName}, id={self.id} start = {format_time(self.start)} duration { self.duration / 60}'

    def __repr__(self):
        return self.__str__()

    def set_sim_start(self, time):
        self.simulated_start = time

    def set_sim_end(self, time):
        self.simulated_end = time

    def get_sim_start(self):
        return self.simulated_start

    def get_sim_end(self):
        return self.simulated_end

    def __hash__(self):
        return self.id



    def set_ready_time(self, value):
        self.ready_time = value


class Simulation:
    nbWeeksRoster = 1
    rosterCycleDays = 7 * nbWeeksRoster
    rosterDayCycleInSeconds = rosterCycleDays * day_in_seconds
    totalCycles = 3

    eventsNoUseResource = []
    batchEvents = []
    batchWorkerMark = []
    singleResoruceEvents = []
    allCases = {}
    allEvents = {}
    resourceData = {}
    dynmaicResoruceData = {}
    rosterResoruceData = {}
    eventResourceQuery = {}
    dynamicEventResourceQuery = {}
    eventSkillSet = {}
    dynamicEvetnSkillSet = {}
    rosterEventSkillSet = {}
    queueStat = {}

    timeWindowMaps = []
    # key by day, value is array of shift window, note the array idx is not same is timeIndex, timeIndex is the starting hour of that shift
    dayTimeWindowMaps = {}
    batchTeamShift = {0: [12, 14], 1: [20, 29]}
    totalJobs = 0
    caseArrivals = {}
    runUntil = None
    nb_days = 40
    timeValueToIndex = {}
    # key by day, timeIndex, event name
    rosterEventResourceQuery = {}
    # key by day, staff name, timeIndex
    dayStaffRostering = {}
    secondsBreakPerShiftPeriod = 10 * 60 # 45 mins per hour, 15 mins break
    batchDebug = []
    breakDebug = []



    def __init__(self, isLocalSim = True):
        self.isLocalSim = isLocalSim
        self.nonRosterResourceGroups = []
        self.nonRosterResource = {}
        self.queueMonitor = {}
        self.priority_res = {}
        self.env = simpy.Environment()
    @classmethod
    def clear(cls):
        cls.eventsNoUseResource = []
        cls.batchEvents = []
        cls.batchWorkerMark = []
        cls.singleResoruceEvents = []
        cls.allCases = {}
        cls.allEvents = {}
        cls.resourceData = {}
        cls.dynmaicResoruceData = {}
        cls.rosterResoruceData = {}
        cls.eventResourceQuery = {}
        cls.dynamicEventResourceQuery = {}
        cls.rosterEventResourceQuery = {}
        cls.eventSkillSet = {}
        cls.dynamicEvetnSkillSet = {}
        cls.rosterEventSkillSet = {}
        cls.timeWindowMaps = []
        cls.dayTimeWindowMaps = {}
        cls.timeValueToIndex = {}
        cls.totalJobs = 0
        cls.caseArrivals = {}
        cls.runUntil = None
        cls.dayStaffRostering = {}
        cls.queueStat = {}
        cls.timeWindowMaps = []
        # key by day, value is array of shift window, note the array idx is not same is timeIndex, timeIndex is the starting hour of that shift
        cls.dayTimeWindowMaps = {}
        cls.batchTeamShift = {0: [12, 14], 1: [20, 29]}
        cls.totalJobs = 0
        cls.caseArrivals = {}
        cls.runUntil = None
        cls.nb_days = 40
        cls.timeValueToIndex = {}
        # key by day, timeIndex, event name
        cls.rosterEventResourceQuery = {}
        # key by day, staff name, timeIndex
        cls.dayStaffRostering = {}

        cls.batchDebug = []
        cls.breakDebug = []

    def getCurrentDaySec(self):
        currentDayInSec = self.env.now // day_in_seconds * day_in_seconds
        return currentDayInSec

    def set_batch_data(self, batchEvents = [], batchWorkerMark = [], singleResourceEvent = [], eventsNoUseResource = [], nonRosterResGroups = []):
        self.batchEvents = batchEvents
        self.batchWorkerMark = batchWorkerMark
        self.singleResoruceEvents =  singleResourceEvent
        self.eventsNoUseResource = eventsNoUseResource
        self.nonRosterResourceGroups = nonRosterResGroups

    def set_roster(self, rosterData):
        self.rosterData = rosterData
        weekDayList = self.rosterData['roster_week']
        finalSimDay = SimSetting.start_time_sim + timedelta(days=self.runUntil)
        roster_start_day = SimSetting.roster_start_time
        # make sure my roster break def go over the last day of sim
        rosterDays = (finalSimDay - roster_start_day).days

        self.rosterCycleDays = len(weekDayList) * 7
        self.totalCycles = math.ceil((rosterDays / self.rosterCycleDays))

    def create_resource_from_roster(self, rosterData):
        staffRosterDict, df, weekDayList = rosterData['roster_dict'], rosterData['roster_df'], rosterData['roster_week']

        nextRosterCycleStart = SimSetting.start_time_sim + timedelta(days=self.rosterCycleDays)
        print(f'next roster cycle start {nextRosterCycleStart}')
        # current_stage_workers = WorkerCollection(tag)
        # for i in range(capacity):
        #    current_stage_workers.add(Worker(i, shift_start, shift_end, members[i]))
        # teamResoruces[day][timeIndex][tag] = MyPriorityResource(self.env, current_stage_workers, tag)

        teams = {}  # key by stage, staff on the same stage belong to the same time

        for staffName, staffInfo in staffRosterDict.items():
            staffInfo: StaffShift = staffInfo
            resource_group = staffInfo.stage
            if resource_group not in teams:
                teams[resource_group] = {}
            teams[resource_group][staffName] = staffInfo
        for resource_group in teams.keys():
            #debug_output(f'nb staff, {resource_group}, {len(teams[resource_group])}')
            current_stage_workers = WorkerCollection(resource_group)
            counter = 0
            for staffInfo in teams[resource_group].values():
                staffInfo: StaffShift = staffInfo

                #debug_output(f'add {staffInfo.name} to stage {resource_group}')
                current_stage_workers.add(Worker(id=counter, start=0, end=0, name=staffInfo.name))
                counter += 1
            self.priority_res[resource_group] = MyPriorityResource(self.env, current_stage_workers, resource_group)
            self.queueMonitor[resource_group] = []
        # key is stage, value is priority resource, worker has the name
        for key, value in self.priority_res.items():
            simResource: MyPriorityResource = value
            #debug_output(f'{key} has {simResource.get_workers()}')
            # nbCycles = math.ceil(self.nb_days * day_in_seconds / self.rosterDayCycleInSeconds)

            # first sleep until the first cycle begin
            # sleepTimeUntilFirstCycle = day * day_in_seconds + shift_start * seconds_per_hour
            capacity = simResource.capacity
            workers = simResource.get_workers()
            for i in range(capacity):
                worker: Worker = workers[i]
                #debug_output(f'worker is {worker} with info {staffRosterDict[worker.name]}')
                staffInfo: StaffShift = staffRosterDict[worker.name]

                # this is shifrt for the first cycle, move on to next cycle
                for currentCycle in range(self.totalCycles):
                    sortedTimekey = sorted(list(staffInfo.shifts.keys()))
                    for idx, timeKey in enumerate(sortedTimekey):
                        shift = staffInfo.shifts[timeKey]

                        year, month, day, shift_start, shift_end = self.get_datetime_from_shift(shift)

                        currentShiftStartTime = shift_start + timedelta(
                            days=currentCycle * self.rosterCycleDays)

                        currentShiftEndTime = shift_end + timedelta(days=currentCycle * self.rosterCycleDays)

                        currentShiftStartTimeFromSimStartInSeconds = (
                                    currentShiftStartTime - SimSetting.start_time_sim).total_seconds()
                        currentShiftEndTimeFromSimStartInSeconds = (
                                    currentShiftEndTime - SimSetting.start_time_sim).total_seconds()
                        if (idx == 0 and currentCycle == 0):
                            # first break, start break after start break time - sim atart time

                            durationBeforeStart = currentShiftStartTimeFromSimStartInSeconds
                            #debug_output(f'start at {format_time(currentDayInSeconds + durationBeforeStart)} end at '
                            #             f'{format_time(currentShiftEndTimeFromSimStartInSeconds)}')
                            self.env.process(self.staff_break_process(simResource, stage=key,
                                                                      worker_id=i, start=0,
                                                                      duration=durationBeforeStart))

                        if idx < len(sortedTimekey) - 1:

                            nextShiftStart = staffInfo.shifts[sortedTimekey[idx + 1]]

                            year, month, day, shift_start, shift_end = self.get_datetime_from_shift(nextShiftStart)

                            nextShiftStartTime = shift_start + timedelta(days=currentCycle * self.rosterCycleDays)
                            nextShiftStartTimeFromSimStartInSeconds = (nextShiftStartTime - SimSetting.start_time_sim).total_seconds()
                            durationBeforeNextShiftStart = nextShiftStartTimeFromSimStartInSeconds - currentShiftEndTimeFromSimStartInSeconds
                            #debug_output(
                            #    f'break at last shift at {format_time(currentShiftEndTimeFromSimStartInSeconds)} end at '
                            #    f'{format_time(currentShiftEndTimeFromSimStartInSeconds + durationBeforeNextShiftStart)}')
                            self.env.process(self.staff_break_process(simResource, stage=key,
                                                                      worker_id=i,
                                                                      start=currentShiftEndTimeFromSimStartInSeconds,
                                                                      duration=durationBeforeNextShiftStart))
                        elif idx == len(sortedTimekey) - 1 and currentCycle < self.totalCycles - 1:
                            # go back to the beining
                            nextCycleFirstShift = staffInfo.shifts[sortedTimekey[0]]


                            year, month, day, shift_start, shift_end = self.get_datetime_from_shift(nextCycleFirstShift)

                            nextCycleShiftStartTime = shift_start + timedelta(
                                days=(currentCycle + 1) * self.rosterCycleDays)
                            #debug_output(f'next cycle start at {nextCycleShiftStartTime}')
                            durationBeforeNextShiftStart = (
                                        nextCycleShiftStartTime - currentShiftEndTime).total_seconds()
                            #debug_output(
                            #    f'break at last shift at {format_time(currentShiftEndTimeFromSimStartInSeconds)} end at '
                            #    f'{format_time(currentShiftEndTimeFromSimStartInSeconds + durationBeforeNextShiftStart)}')
                            self.env.process(self.staff_break_process(simResource, stage=key,
                                                                      worker_id=i,
                                                                      start=currentShiftEndTimeFromSimStartInSeconds,
                                                                      duration=durationBeforeNextShiftStart))


    def isEventNoUseResource(self, eventName):
        return eventName in self.eventsNoUseResource


    def set_run_until(self, days = 30):
        self.runUntil = days
        self.nb_days = days + 8



    def find_unique_server(self, resourceData):
        eventResource = {}
        for tran, rSet in self.eventSkillSet.items():
            eventResource[tran] = {}
            singleUse = []
            multiUse = []
            for r in rSet:
                if len(resourceData[r]) == 1:
                    # this resource can only do this event
                    singleUse.append(r)
                else:
                    multiUse.append(r)
            eventResource[tran][singleUseTag] = singleUse
            eventResource[tran][multiUseTag] = multiUse
        return eventResource

    def find_dynamic_teams(self, resourceData):
        teams = {}

        for timeIndex, res in resourceData.items():
            teams[timeIndex] = {}
            processedMembers = []
            for r, trans in res.items():
                if r in processedMembers: continue
                processedMembers.append(r)
                sameTeam = [r]
                for s, sTrans in res.items():
                    if r == s: continue
                    if trans == sTrans:
                        sameTeam.append(s)
                        processedMembers.append(s)
                teams[timeIndex][r] = team_type(teamMember=sameTeam, events=trans)

        return teams


    def find_teams(self, resourceData):
        teams = {}
        processedMembers = []
        for r, trans in resourceData.items():
            if r in processedMembers: continue
            processedMembers.append(r)
            sameTeam = [r]
            for s, sTrans in resourceData.items():

                if r == s: continue
                if trans == sTrans:
                    sameTeam.append(s)
                    processedMembers.append(s)

            # now I now who I can work together as a team coz we do the same events
            # key is the first member in that team
            teams[r] = team_type(teamMember=sameTeam, events=trans)
        debug_output(f'teams')
        return teams

    def timeWindow_row_process(self, row):
        timeIndex = row.timeIndex
        start = row.start
        end = row.end
        self.timeWindowMaps.append(timeWindow_type(timeIndex=timeIndex, start=start, end=end))
        self.timeWindowMaps = sorted(self.timeWindowMaps, key=lambda x: x.timeIndex)





    def dynamic_resource_row_process(self, row, resourceData):

        timeIndex = row.timeIndex
        if timeIndex not in resourceData:
            resourceData[timeIndex] = {}
        transition = row.transition

        resource_set = ast.literal_eval(row.resource_set)
        if timeIndex not in self.dynamicEvetnSkillSet:
            self.dynamicEvetnSkillSet[timeIndex] = {}
        self.dynamicEvetnSkillSet[timeIndex][transition] = resource_set
        # resourceType  = resource_type(transition=transition, capacity=capacity, resource = resource)
        # for each resource, create list of events that require this resource
        for r in resource_set:
            if r not in resourceData[timeIndex]:
                resourceData[timeIndex][r] = []
            if transition not in resourceData[timeIndex][r]:
                resourceData[timeIndex][r].append(transition)
        # group resoruces that only serve unique events into a single resource

    def resource_row_process(self, row, resourceData):
        transition = row.transition
        resource_set = row.resource_set
        # Converting string to list
        if self.isLocalSim:
            resource_set = ast.literal_eval(resource_set)
        self.eventSkillSet[transition] = resource_set
        # resourceType  = resource_type(transition=transition, capacity=capacity, resource = resource)
        # for each resource, create list of events that require this resource
        for r in resource_set:
            if r not in resourceData:
                resourceData[r] = []
            if transition not in resourceData[r]:
                resourceData[r].append(transition)
        # group resoruces that only serve unique events into a single resource


    def break_process(self, env, resource = None, worker_id: int = 0, stage=1, start=0, duration=3):


        expcted_break = env.now + start
        # start is the time duration from simulation time 0
        yield env.timeout(start)

        # at the time to reqeust break, we make the request, this request is put into queue at this moment with high priority
        # this high priorit will move break request to the top of the queue and order by request time
        # when any resoruce is freed, it will call back the queue and start from the first event on the queue
        # becoase all break event have high piroirty, ther will be no job events between them
        # by going through all break events  we make sure the specific worker will take the rest for sure when it is free
        with resource.request(priority=-4, event=BreakEvent(worker_id)) as req:
            yield req
            # I want to know which worker start break

            if 'processing_1000'  in stage and worker_id==0 and (env.now // day_in_seconds >=4) :
                debug_output(f'stage {stage} worker {worker_id} start break at {format_time(env.now)}, expected at {format_time(expcted_break)} finish at {format_time(env.now+duration)}'
                             , show=True)
                self.batchDebug.append([worker_id, format_time(expcted_break),format_time(env.now+duration) ])
            # if I start a break AND I have job request want to interrupt that request
            # I need to know job_id
            # worker may work a bit of of time, so he may not get the request at the exact time
            # make sure he start the correct time next day

            actualDuration = duration - (env.now - expcted_break)
            if 'processing'  in stage:
                self.breakDebug.append([f'{stage}_{worker_id}', format_time(env.now), format_time(expcted_break), format_time(env.now+actualDuration)])
            yield env.timeout(actualDuration)
        if 'processing_1000'  in stage and (env.now // day_in_seconds >=5):
            debug_output(f'stage {stage} worker {worker_id}  break finish at {format_time(env.now)}', show=True)

        # if it is a lunch stage, finish break, I want to pass all the rqeusts for this resource to night stage, meaning inrurrut


    def create_batch_resource(self, env: simpy.Environment, team, teamResoruces, nbDays):
        # break this team by half
        events = team.events
        members = team.teamMember
        firstTeam = members[0: len(members) // 2]
        secondTeam = members[len(members) // 2:]
        batchTeams = [firstTeam, secondTeam]

        for idx, t in enumerate(batchTeams):
            tag = idx
            shift_start = self.batchTeamShift[idx][0]
            shift_end = self.batchTeamShift[idx][1]
            current_stage_workers = WorkerCollection(tag, shift_start, shift_end)
            capacity = len(t)

            for i in range(capacity):
                current_stage_workers.add(Worker(i, shift_start, shift_end, t[i]))
            teamResoruces[tag] = MyPriorityResource(env, current_stage_workers, tag)
            self.define_break(capacity, env, nbDays, shift_end, shift_start, teamResoruces[tag], tag)
            for event in events:
                if event not in self.eventResourceQuery:
                    self.eventResourceQuery[event] = {}
                self.eventResourceQuery[event][tag] = teamResoruces[tag]


    def create_resource_from_team(self, env: simpy.Environment, teams):
        nbDays = self.nb_days + 1
        shift_start = 8
        shift_end = 18

        teamResoruces = {}

        # for each event I need to get the correct resoruces to query
        for firstMember, team in teams.items():
            events = team.events
            members = team.teamMember
            capacity = len(members)
            tag = firstMember
            if (tag in self.batchWorkerMark):
                self.create_batch_resource(env, team, teamResoruces, nbDays)
                continue
            current_stage_workers = WorkerCollection(tag)
            for i in range(capacity):
                current_stage_workers.add(Worker(i, shift_start, shift_end, members[i]))

            teamResoruces[tag] = MyPriorityResource(env, current_stage_workers, tag)
            self.define_break(capacity, env, nbDays, shift_end, shift_start, teamResoruces[tag], tag)

            for event in events:
                if event not in self.eventResourceQuery:
                    self.eventResourceQuery[event] = {}
                self.eventResourceQuery[event][tag] = teamResoruces[tag]


    def define_break(self, capacity, env, nbDays, shift_end, shift_start, simResource, key):
        for day in range(nbDays):
            for i in range(capacity):
                dayOfWeek = day % 7

                if (day == 0):
                    env.process(self.break_process(env, simResource, stage=key, worker_id=i, start=0,
                                              duration=shift_start * seconds_per_hour))
                    env.process(self.break_process(env, simResource, stage=key, worker_id=i,
                                              start=day * day_in_seconds + shift_end * seconds_per_hour,
                                              duration=(shift_start + 24 - shift_end) * seconds_per_hour))
                #elif dayOfWeek == 4:
                    # take 2 days break
                #    env.process(self.break_process(env, simResource, stage=key, worker_id=i,
                #                              start=day * day_in_seconds + shift_end * seconds_per_hour,
                #                              duration=3 * day_in_seconds - (shift_end - shift_start) * seconds_per_hour))
                #elif dayOfWeek >4:
                #    continue
                else:
                    env.process(self.break_process(env, simResource, stage=key, worker_id=i,
                                              start=day * day_in_seconds + shift_end * seconds_per_hour,
                                              duration=(shift_start + 24 - shift_end) * seconds_per_hour))

    def singleResoruceProcess(self, env: simpy.Environment, eventsArray, dependentEventDict):
        allFinished = []
        for key, finishSignals in eventsArray.items():
            # to start reqeust for batching, all grouped events must be ready
            for f in finishSignals:
                allFinished.append(f)

        # print(event)
        debug_output(f'try do single resoruce at {format_time(env.now)}')
        # for each node, all predessors need to finish before continue
        yield env.all_of(allFinished)
        debug_output(f'single, parents are fnished')
        debug_output(f'want resource do this task  batching at time {format_time(env.now)}')

        # these events require the same resource worker, so just do them on by one then release the resoruce
        event = key
        transitionName = event.group
        events = [e for e in eventsArray.keys()]
        priority = 0
        # get skill set resurces
        requestOrEvents = []
        resourceDict = {}
        queryResources = self.eventResourceQuery[transitionName]
        foundFree = False
        freeResource = None
        req = None
        for key, r in queryResources.items():
            requestedResource: MyPriorityResource = r
            if requestedResource.at_least_one_free():
                # no need to reqeust more resources
                foundFree = True
                freeResource = requestedResource
                break
        if not foundFree:
            for key, r in queryResources.items():
                requestedResource: MyPriorityResource = r

                newRequest = requestedResource.request(event=NewJobEvent(worker_id=None, job_id=event.id),
                                                       priority=priority)
                resourceDict[newRequest] = requestedResource
                requestOrEvents.append(newRequest)
            req = yield env.any_of(requestOrEvents)
            # return a dict with reqeust as key, return value is value
            totalTriggered = 0
            # get the first free worker0 = {MyPriorityRequest} priority is 0 for job job 7 reqeust for resource resource Oddo2
            freeResourceRequest: MyPriorityRequest = None
            for e in req.events:
                e1: MyPriorityRequest = e
                if e1.triggered:
                    totalTriggered = totalTriggered + 1
                    freeResourceRequest = e1
                    break
            for e in requestOrEvents:
                e1: MyPriorityRequest = e
                if e1 != freeResourceRequest and not e1.triggered:
                    # in the queue list
                    e1.cancel()
                elif e1 != freeResourceRequest:
                    assert (e1.triggered)
                    # remove from the user list
                    resourceDict[e1].release(e1)
            # print('total free workers is ', totalTriggered)
            w: Worker = req[freeResourceRequest]
            for event in events:
                debug_output(f' I can do this node {event} at time {format_time(env.now)} by {w}')
                event.set_resource(w.name)
                event.set_sim_start(env.now)
                duration = event.duration
                yield env.timeout(duration)
                debug_output(f'finish this task {event} at time {env.now}')
                event.set_sim_end(env.now)
                for key, eventList in dependentEventDict.items():
                    if (len(eventList) == 0): continue
                    for parent in eventList:
                        if event.id == parent[0]:
                            debug_output(f' for node {key}, the dependend event is succed {event}')
                            parent[1].succeed()

            resourceDict[freeResourceRequest].release(freeResourceRequest)
            debug_output(f' I can do this node {event} at time {format_time(env.now)} by {w}')
        else:
            req = freeResource.request(event=NewJobEvent(worker_id=None, job_id=event.id), priority=priority)
            w: Worker = yield req

            for event in events:
                debug_output(f' I can do this node {event} at time {format_time(env.now)} by {w}')
                event.set_resource(w.name)
                event.set_sim_start(env.now)
                duration = event.duration
                yield env.timeout(duration)
                debug_output(f'finish this task {event} at time {env.now}')
                event.set_sim_end(env.now)
                for key, eventList in dependentEventDict.items():
                    if (len(eventList) == 0): continue
                    for parent in eventList:
                        if event.id == parent[0]:
                            debug_output(f' for node {key}, the dependend event is succed {event}')
                            parent[1].succeed()
            if (freeResource is not None):
                freeResource.release(req)



    def nodeProcess(self, env: simpy.Environment, event, eventsArray, dependentEventDict):
        if event.is_start:
            # must wait for its arrival
            arrival = self.caseArrivals[event.case_id]
            yield env.timeout(arrival - env.now)
        # print(event)
        debug_output(f'try do {event} at {format_time(env.now)}')
        # for each node, all predessors need to finish before continue
        yield env.all_of(eventsArray[event])
        debug_output(f'{event.id}, parents are fnished, {eventsArray[event]}')
        debug_output(f'want resource do this task {event} at time {format_time(env.now)}')
        # try dellee unused memory
        # del eventsArray[node]
        # del dependentEventDict[node]
        transitionName = event.group
        priority = 0
        # get skill set resurces
        requestOrEvents = []
        resourceDict = {}
        queryResources = self.eventResourceQuery[transitionName]
        foundFree = False
        freeResource = None
        req = None
        for key, r in queryResources.items():
            requestedResource: MyPriorityResource = r
            if requestedResource.at_least_one_free():
                # no need to reqeust more resources
                foundFree = True
                freeResource = requestedResource
                break
        if not foundFree:
            for key, r in queryResources.items():
                requestedResource: MyPriorityResource = r

                newRequest = requestedResource.request(event=NewJobEvent(worker_id=None, job_id=event.id),
                                                       priority=priority)
                resourceDict[newRequest] = requestedResource
                requestOrEvents.append(newRequest)
            req = yield env.any_of(requestOrEvents)
            # return a dict with reqeust as key, return value is value
            totalTriggered = 0
            # get the first free worker0 = {MyPriorityRequest} priority is 0 for job job 7 reqeust for resource resource Oddo2
            freeResourceRequest: MyPriorityRequest = None
            for e in req.events:
                e1: MyPriorityRequest = e
                if e1.triggered:
                    totalTriggered = totalTriggered + 1
                    freeResourceRequest = e1
                    break
            for e in requestOrEvents:
                e1: MyPriorityRequest = e
                if e1 != freeResourceRequest and not e1.triggered:
                    # in the queue list
                    e1.cancel()
                elif e1 != freeResourceRequest:
                    assert (e1.triggered)
                    # remove from the user list
                    resourceDict[e1].release(e1)
            # print('total free workers is ', totalTriggered)
            w: Worker = req[freeResourceRequest]
            event.set_resource(w.name)
            event.set_sim_start(env.now)
            duration = event.duration
            yield env.timeout(duration)
            debug_output(f'finish this task {event} at time {env.now}')
            event.set_sim_end(env.now)

            resourceDict[freeResourceRequest].release(freeResourceRequest)
            debug_output(f' I can do this node {event} at time {format_time(env.now)} by {w}')
        else:
            req = freeResource.request(event=NewJobEvent(worker_id=None, job_id=event.id), priority=priority)
            w: Worker = yield req
            debug_output(f' I can do this node {event} at time {format_time(env.now)} by {w}')
            event.set_resource(w.name)

            event.set_sim_start(env.now)
            duration = event.duration
            yield env.timeout(duration)
            debug_output(f'finish this task {event} at time {format_time(env.now)}')
            event.set_sim_end(env.now)
            if (freeResource is not None):
                freeResource.release(req)
        # if another node depends this node to fnish, notify that node this node is finished
        for key, eventList in dependentEventDict.items():
            if (len(eventList) == 0): continue
            for parent in eventList:
                if event.id == parent[0]:
                    debug_output(f' for node {key}, the dependend event is succed {event}')
                    parent[1].succeed()
        # this request needs to know the associated resource


    def batchProcess(self, env: simpy.Environment, event, eventsArray, dependentEventDict):
        if event.is_start:
            # must wait for its arrival
            arrival = self.caseArrivals[event.case_id]
            yield env.timeout(arrival - env.now)
        # print(event)
        debug_output(f'try do {event} at {format_time(env.now)}')
        # for each node, all predessors need to finish before continue
        yield env.all_of(eventsArray[event])
        debug_output(f'{event.id}, parents are fnished, {eventsArray[event]}')
        debug_output(f'want resource do this task {event} at time {format_time(env.now)}')
        # try dellee unused memory
        # del eventsArray[node]
        # del dependentEventDict[node]
        transitionName = event.group
        priority = 0
        # get skill set resurces
        requestOrEvents = []
        resourceDict = {}
        queryResources = self.eventResourceQuery[transitionName]

        # for batching, I need to find the next avaiable batch by testing my current time
        batch_time = env.now // day_in_seconds * day_in_seconds + 20 * seconds_per_hour
        currentDay = env.now // day_in_seconds * day_in_seconds
        nextBatch: MyPriorityResource = None
        nextBatchStartTime = None
        nextBatchDuration = None
        w: Worker = None
        success = False
        # sort batch resources b start time
        values = []
        for key, value in queryResources.items():
            if event.is_in_batch(key):
                values.append(value)
        values.sort(key=lambda res: res.get_workers().get_shift_start())
        while not success:
            # if I have passed last batch, go back to the fist batch next day
            if (env.now > values[-1].get_workers().get_shift_start() * seconds_per_hour + currentDay):
                currentDay += day_in_seconds
            for value in values:
                res: MyPriorityResource = value
                workers: WorkerCollection = res.get_workers()
                # want the next earliest possible batch
                # it does not have to be the same day
                # for example after second batch, I should look for the first batch next day
                # I have an array of batch resoures, loop through for the next batch

                workerStart = workers.get_shift_start() * seconds_per_hour + currentDay
                if env.now <= workerStart:
                    nextBatch = res
                    nextBatchStartTime = workerStart
                    nextBatchDuration = workers.get_duration_in_hours() * seconds_per_hour
                    break
            debug_output(f'next batch for {event} is {format_time(nextBatchStartTime)} ')
            quitEvent = env.timeout(nextBatchStartTime - env.now + 1)
            newRequest = nextBatch.request(event=NewJobEvent(worker_id=None, job_id=event.id), priority=priority)
            result = yield newRequest | quitEvent
            if quitEvent in result:
                # too much waiting, quit
                success = False
                debug_output(f'quit wait at {format_time(env.now)} for {event.case_id}')
                newRequest.cancel()
                # request night batch
            else:
                # job is done
                success = True
                w = result[newRequest]

        debug_output(f' I can do this node {event} at time {format_time(env.now)} by {w}')

        event.set_resource(w.name)
        event.set_sim_start(env.now)
        yield env.timeout(nextBatchDuration)
        if (event.case_id == 'Case 14' and event.group == 'trans_10'):
            print('im here')
        debug_output(f'finish this task {event} at time {format_time(env.now)}')
        event.set_sim_end(env.now)
        nextBatch.release(newRequest)
        # if another node depends this node to fnish, notify that node this node is finished
        for key, eventList in dependentEventDict.items():
            if (len(eventList) == 0): continue
            for parent in eventList:
                if event.id == parent[0]:
                    debug_output(f' for node {key}, the dependend event is succed {event}')
                    parent[1].succeed()
        # this request needs to know the associated resource



    def generate_dependent_from_dict(self, case_id, env):
        dependentEventDict = {}
        notifyChildren = {}
        eventsArray = {}
        events: list[Event] = self.allCases[case_id]
        for event in events:
            # debug_output('parents')
            dependentEventDict[event] = []
            eventsArray[event] = []
            for parent in event.parents:
                # debug_output(f'{allEvents[parent]}')
                finishEvent = env.event()

                dependentEventDict[event].append([parent, finishEvent])
                eventsArray[event].append(finishEvent)
            # what do I do with children?
        # build a dict of parent finish events for every event of this case
        # how do I notify the children

        return eventsArray, dependentEventDict

    def get_datetime_from_roster(self, shift):
        year = shift[0]
        month = shift[1]
        day = shift[2]
        start_hour = shift[3].hour
        start_min = shift[3].minute
        end_hour = shift[4].hour
        endt_min = shift[4].minute
        return day, end_hour, endt_min, month, start_hour, start_min, year

    def get_datetime_from_shift(self, shift):
        year = shift[0]
        month = shift[1]
        day = shift[2]
        shift_start = shift[3]
        shift_end = shift[4]
        return year, month, day, shift_start, shift_end

    def staff_break_process(self, resource=None, worker_id: int = 0, stage=1, start=0, duration=3):

        expcted_break = self.env.now + start
        # start is the time duration from simulation time 0
        yield self.env.timeout(start)
        debug_output(f'start asking break for {stage} at {format_time(start)} for {worker_id}')

        # at the time to reqeust break, we make the request, this request is put into queue at this moment with high priority
        # this high priorit will move break request to the top of the queue and order by request time
        # when any resoruce is freed, it will call back the queue and start from the first event on the queue
        # becoase all break event have high piroirty, ther will be no job events between them
        # by going through all break events  we make sure the specific worker will take the rest for sure when it is free
        with resource.request(priority=-4, event=BreakEvent(worker_id)) as req:
            yield req
            # I want to know which worker start break

            debug_output(f'stage {stage} worker {worker_id} start break at {format_time(self.env.now)}, expected at {format_time(expcted_break)} finish at {format_time(self.env.now + duration)}')
            # self.batchDebug.append([worker_id, format_time(expcted_break),format_time(self.env.now+duration) ])
            # if I start a break AND I have job request want to interrupt that request
            # I need to know job_id
            # worker may work a bit of of time, so he may not get the request at the exact time
            # make sure he start the correct time next day

            actualDuration = duration - (self.env.now - expcted_break)

            self.breakDebug.append([f'{stage}_{worker_id}', format_time(self.env.now), format_time(expcted_break),
                                    format_time(self.env.now + actualDuration)])
            yield self.env.timeout(actualDuration)

    def load_next_job_set_by_day(self, period=step, job_bin=None):
        global totalJobs
        load_next = True
        last_day = (self.nb_days + 1) * day_in_seconds
        while load_next:
            if self.env.now >= last_day:
                return
            at_least_one = False
            job_included = []

            current_time = SimSetting.start_time_sim + timedelta(seconds=self.env.now)

            dayIndex = date(year=current_time.year, month=current_time.month, day=current_time.day)

            hasDataForToday = dayIndex in job_bin
            if hasDataForToday:
                jobs = job_bin[dayIndex]
                new_cases = []
                jobs.apply(self.event_row_process_new_cases, new_cases=new_cases, axis=1)
                self.totalJobs = self.totalJobs + len(new_cases)
                for case_id in new_cases:

                    if True:
                        # G = create_dependcy_for_case(case_id)
                        job_included.append(case_id)
                        # eventsArray, dependentEventDict = generate_dependent_signals(G)
                        eventsArray, dependentEventDict = self.generate_dependent_from_dict(case_id, self.env)

                        batchArray = {}
                        singleResourceEventsArray = {}
                        keysToDelete = []
                        for key in eventsArray:
                            if key.group in self.batchEvents:
                                batchArray[key] = eventsArray[key]
                                keysToDelete.append(key)
                            for name in self.singleResoruceEvents:
                                if name in key.group:
                                    singleResourceEventsArray[key] = eventsArray[key]
                                    keysToDelete.append(key)

                        # grop events into batch jobs and non batch jobs

                        for key in eventsArray:
                            if key not in keysToDelete:
                                self.env.process(self.nodeProcess(key, eventsArray, dependentEventDict))
                        if (len(batchArray) > 0):
                            self.env.process(self.batchProcess(batchArray, dependentEventDict))
                        if len(singleResourceEventsArray) > 0:
                            self.env.process(self.singleResoruceProcess(singleResourceEventsArray, dependentEventDict))
                        at_least_one = True

                if not at_least_one and self.env.now >= last_day:
                    load_next = False
                print(f'load jobs {len(job_included)} at {format_time(self.env.now)}')

            yield self.env.timeout(step)
            period = period + step

    def create_non_roster_res(self):
        for key, value in self.nonRosterEventResource.items():
            tag = key
            self.nonRosterResource[tag] =  simpy.PriorityResource(self.env, capacity=value)

    def GetFreeRosterResource(self, event, priority):

        freeResource = self.priority_res[event.group]
        freeResourceRequest = freeResource.request(event=NewJobEvent(worker_id=None, job_id=event.id),
                                                   priority=priority)
        w: Worker = yield freeResourceRequest
        return freeResource, freeResourceRequest, w



