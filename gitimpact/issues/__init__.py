class BugTrackerIssuesGenerator():
    def __init__(self):
        pass

    def create_issue(self, id):
        return BugTrackerIssue(id)

class BugTrackerIssue():
    def __init__(self, id):
        self.id = id

    def load_data(self):
        pass

    def fill_data(self, bugtracker_issue):
        pass