# coding=utf-8
import os

from git import Repo

from gitimpact.core import ImpactAnalyser
from gitimpact.core.addons.git import GitRepoLoader, GitUtils
from gitimpact.core.calc.calc import NPNewCosineSimilarityCalculator

from reprint import output
from time import sleep


def calc_status_impact(repo_path, results_count=10, commits_depth=50, interval=1, debug=False):
    repo = Repo.init(repo_path)
    git_worker = GitRepoLoader(repo)
    ia = ImpactAnalyser()

    calc_type = NPNewCosineSimilarityCalculator

    all_items = git_worker.load_commits(commits_depth=commits_depth)
    # all_issue_items = group_by_issues(all_items)

    print(u"Git graph loaded")

    with output(output_type="list", initial_len=results_count, interval=0) as output_list:
        while True:
            test_item = GitUtils.repo_status_to_container(repo)
            # print(u"Original item\n%s\n" % (test_item.name))

            results = ia.impact_analysis_for_commit(test_item, all_items, results_count=results_count,
                                                    debug=debug,
                                                    calculator_type=calc_type)

            result_strings = []
            for container, distance in results:
                impact = 1.0 - distance
                if distance >= 1.0:
                    continue
                result_strings.append(u'%s\t%f' % (container.name, impact))
                # result_strings.append(u'%s [%s] – %f' % (container.name, container.message.split('\n')[0], distance))
            result_strings.extend([u''] * max(0, results_count - len(result_strings)))

            for idx, result_string in enumerate(result_strings):
                output_list[idx] = result_string

            sleep(interval)


def main(app_options):
    params = {
        "repo_path": app_options.repo_path,
        "results_count": int(app_options.results_count),
        "commits_depth": int(app_options.commits_depth),
        "interval": int(app_options.interval),
        "debug": app_options.debug
    }
    calc_status_impact(**params)
