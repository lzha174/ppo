import math
from datetime import datetime, timedelta, date
#import plotly.express as px
import pandas as pd

seconds_per_hour = 60 * 60

day_in_seconds = seconds_per_hour * 24
step = day_in_seconds

def setup_sim_setting(backlogWeek=1, rosterWeekBlockSpan=3, jobdays = 5, year = 2021, month = 5, day = 3):
    start_time_sim = datetime(year, month, day, 0, 0, 0)
    roster_start_time = start_time_sim + timedelta(days=backlogWeek * 7)
    weekBlockStart = roster_start_time.isocalendar()[1]

    dayMapping = {}
    jobTotalDays = jobdays
    rosterTotalDays = rosterWeekBlockSpan * 7
    for i in range(rosterTotalDays):
        newDate = roster_start_time + timedelta(days=i)
        dayMapping[newDate] = i

    # weekDayList is defined by weekBlockSpan, totalDays are the total number of days across this span
    rosterWeekDayList = []
    for i in range(rosterWeekBlockSpan):
        # a list is a tuple of monday to sun in each week
        weekStart = roster_start_time + timedelta(days=i * 7)
        weekEnd = roster_start_time + timedelta(days=(i + 1) * 7 - 1)
        rosterWeekDayList.append((weekStart, weekEnd))

    jobWeekDayList = []
    for i in range(rosterWeekBlockSpan + backlogWeek):
        # a list is a tuple of monday to sun in each week
        weekStart = start_time_sim + timedelta(days=i * 7)
        weekEnd = start_time_sim + timedelta(days=(i + 1) * 7 - 1)
        jobWeekDayList.append((weekStart, weekEnd))

    print('job week', jobWeekDayList)
    return start_time_sim, roster_start_time, rosterWeekDayList, jobWeekDayList, jobTotalDays, rosterTotalDays, weekBlockStart, rosterWeekBlockSpan, dayMapping


class SimSetting():
    start_time_sim = None
    roster_start_time= None
    rosterWeekDayList= None
    jobWeekDayList= None
    jobTotalDays= None
    rosterTotalDays= None
    weekBlockStart= None
    rosterWeekBlockSpan= None
    dayMapping= None
    backlogWeek = None
    @classmethod
    def setup(cls, backlogWeek=1, rosterWeekBlockSpan=3, jobdays = 5, year=2021, month=5, day=3):
        cls.start_time_sim, cls.roster_start_time, cls.rosterWeekDayList, cls.jobWeekDayList, cls.jobTotalDays, \
        cls.rosterTotalDays, cls.weekBlockStart, cls.rosterWeekBlockSpan, cls.dayMapping = setup_sim_setting(
            backlogWeek=backlogWeek, rosterWeekBlockSpan=rosterWeekBlockSpan, jobdays = jobdays, year=year, month=month, day=day)
        cls.backlogWeek = backlogWeek


def getDateTime(n, out_date_format_str='%m/%d/%Y %H:%M:%S'):
    n = int(n)
    final_time = SimSetting.start_time_sim + timedelta(seconds=n)

    final_time_str = final_time.strftime(out_date_format_str)
    return final_time_str

def getDateTimeObject(n):
    final_time = SimSetting.start_time_sim + timedelta(seconds=n)
    return final_time

def format_time(n=15):
    return getDateTime(n)


def covert_df_job_pattern(job_array):
    jobDistribution = JobPattern()
    for v in job_array:
        jobDistribution.upsert_job_distribution(v[0].year, v[0].month, v[0].day, v[0].hour, v[1])
    return jobDistribution


def df_roster_row(row, staffRosterDict):
    name = row.Task
    stage = row.Resource
    start = row.Start
    finish = row.Finish

    year = start.year
    month = start.month
    day = start.day
    startHour = start.hour
    startMin = float(start.minute) / 60
    startHour = startHour + startMin

    endMin = float(finish.minute) / 60
    endHour = finish.hour + endMin
    duration = endHour - startHour

    if name not in staffRosterDict:
        staff = StaffShift(name=name, stage=stage)
        staffRosterDict[name] = staff
    else:
        staff = staffRosterDict[name]
    staff.upsert_shift(year, month, day, start, duration)


class StaffShift():
    def __init__(self, name, stage):
        self.name = name
        self.stage = stage
        self.shifts = {}
        self.first_shift = None

    def get_day_shift(self, year, month, day):
        if self.shifts.get((year, month, day)) != None:
            return self.shifts[year, month, day]
        return None

    def upsert_shift(self, year, month, day, start: datetime, duration, stage=None):
        # start is in datetime, duration is in decimal?
        if stage is not None:
            self.stage = stage
        durationHour = math.floor(duration)
        durationMin = (duration - durationHour) * 60
        finish = start + timedelta(hours=durationHour, minutes=durationMin)
        self.shifts[(year, month, day)] = [year, month, day, start, finish]
        if len(self.shifts) == 1:
            self.first_shift = [year, month, day, start, finish]

    def get_first_shift(self):
        return self.first_shift

    def __str__(self):
        return f' {self.stage}, {self.shifts}'

    def __repr__(self):
        return self.__str__()


# specifiy nb of jobs per hour
class JobPattern():
    def __init__(self):
        self.job_data = {}

    def upsert_job_distribution(self, year, month, day, hour, weight):
        self.job_data[year, month, day, hour] = weight


def add_shift(i, staff, start: datetime, duration):
    newDate = SimSetting.roster_start_time + timedelta(days=i)
    staff.upsert_shift(newDate.year, newDate.month, newDate.day, start, duration)


def rosterDF(staffRosterDict):
    dfData = []
    for staffName, value in staffRosterDict.items():
        value: StaffShift = value
        for shift, shiftData in value.shifts.items():
            # need to deal with over time for shfit ending
            # covert hour decimal to hour, minitues
            vStart: datetime = shiftData[3]
            vEnd: datetime = shiftData[4]
            dfData.append(dict(Task=staffName, Start=vStart, Finish=vEnd, Resource=value.stage))
    df = pd.DataFrame(dfData)
    return df


def group_roster_by_date(staffRosterDict):
    grouped_roster = {}  # key is stage first, then subkey is date, value is staff names
    for name, shiftData in staffRosterDict.items():
        shiftData: StaffShift = shiftData
        stage = shiftData.stage
        if stage not in grouped_roster:
            grouped_roster[stage] = {}
        for key, shift in shiftData.shifts.items():
            shift_start_date = datetime(year=key[0], month=key[1], day=key[2])
            if shift_start_date not in grouped_roster[stage]:
                grouped_roster[stage][shift_start_date] = []
            grouped_roster[stage][shift_start_date].append(name)
    return grouped_roster


def rostertoCoverage(staffRosterDict=None, stages=[]):
    # get coverae for each half hour

    results = []

    weekBlockEnd = SimSetting.weekBlockStart + SimSetting.rosterWeekBlockSpan - 1

    weeks = [i for i in range(SimSetting.weekBlockStart, weekBlockEnd + 1)]
    timeMapping = {}
    timeMappingReverse = {}
    for timeIndx, timeValue in enumerate(range(0, 48)):
        timeMapping[timeIndx] = 0.5 * timeValue
        timeMappingReverse[0.5 * timeValue] = timeIndx

    days = [0, 1, 2, 3, 4, 5, 6]
    print('weeks are', weeks)
    # build a dict for coverage
    coverageDict = {}
    # key is week, day, timeIndex (start from 0)
    for idx, week in enumerate(weeks):
        dayCounterOffset = 7 * idx
        for day in days:
            for timeIndx, timeValue in enumerate(range(0, 48)):
                # print(timeIndx, 0.5 * timeValue)
                coverageDict[day + dayCounterOffset, timeIndx] = {}
                for stage in stages:
                    coverageDict[day + dayCounterOffset, timeIndx][stage] = 0

    # print(SimSetting.dayMapping)
    for key, value in coverageDict.items():
        # key is [dayIndex, timeIndx]
        # value is {stage: coverage}
        # given a key, accuulate all values for that key
        for s, sValue in staffRosterDict.items():
            sValue: StaffShift = sValue
            shifts = sValue.shifts
            for shiftKey, shiftValue in shifts.items():
                start_year = shiftKey[0]
                start_month = shiftKey[1]
                start_day = shiftKey[2]
                end_year = shiftValue[4].year
                end_month = shiftValue[4].month
                end_day = shiftValue[4].day
                # find the dayIndex for this shift
                start_dayIndex = SimSetting.dayMapping[datetime(year=start_year, month=start_month, day=start_day)]
                end_dayIndex = SimSetting.dayMapping[datetime(year=end_year, month=end_month, day=end_day)]
                if start_dayIndex != key[0] and end_dayIndex != key[0]: continue
                stage = sValue.stage

                startHour = shiftValue[3].hour
                endHour = shiftValue[4].hour
                a = shiftValue[3].strftime('%y-%m-%d')

                b = shiftValue[4].strftime('%y-%m-%d')
                if b > a:
                    # overnight shift, for starting day, assume end at 24
                    # for next day, assume start at 0
                    if start_dayIndex == key[0]:
                        endHour = 24
                    elif end_dayIndex == key[0]:
                        startHour = 0

                coverageInterval = pd.Interval(left=timeMapping[key[1]], right=timeMapping[key[1]] + 0.5,
                                               closed='left')
                shiftInterval = pd.Interval(left=startHour, right=endHour, closed='left')
                if shiftInterval.overlaps(coverageInterval):
                    value[stage] = value[stage] + 1
        # print(coverageDict)
    coverageData = []
    for key, value in coverageDict.items():
        data = []
        for stage in stages:
            nbStaff = value[stage]
            data.append(nbStaff)
            if nbStaff > 0:
                staffList = []

                for i in range(nbStaff):
                    staffList.append(f'{stage}_{i}')
                results.append([key[0], key[1], stage, staffList])
        coverageData.append([key[0], key[1]] + data)
        if True:
            processingList = [f'processing_{i}' for i in range(2400)]

            results.append([key[0], key[1], 'processing', processingList])

    coverDataLog = pd.DataFrame(coverageData, columns=['day', 'timeIndex'] + stages)
    coverDataLog = coverDataLog.sort_values(["day", "timeIndex"], ascending=(True, True))
    # print(coverDataLog)

    dataLog = pd.DataFrame(results, columns=['day', 'timeIndex', 'transition', 'resource_set'])
    dataLog = dataLog.sort_values(["day", "timeIndex", "transition"], ascending=(True, True, True))
    dataLog.to_csv('day_roster.csv', index=False)
    # calcualte average maek span

    results.clear()
    for d, dayIndex in SimSetting.dayMapping.items():
        for timeIndex, timeValue in timeMapping.items():
            results.append([dayIndex, timeIndex, timeValue, timeValue + 0.5])
    dayTimeLog = pd.DataFrame(results, columns=['day', 'timeIndex', 'start', 'end'])
    dayTimeLog = dayTimeLog.sort_values(["day", "timeIndex"], ascending=(True, True))
    dayTimeLog.to_csv('dayTimeMap.csv', index=False)
    return dataLog, dayTimeLog, coverDataLog


def load_roster_df(roster_df):
    print(roster_df.info())
    staffRosterDict = {}
    roster_df.apply(df_roster_row, staffRosterDict=staffRosterDict, axis=1)

    return staffRosterDict
