# coding=utf-8
import os

from git import Repo

from gitimpact.core import ImpactAnalyser
from gitimpact.core.addons.git import GitRepoLoader, GitUtils
from gitimpact.core.calc.calc import NPNewCosineSimilarityCalculator

CONFIG_FILENAME = ".gitimpact.yml"
CONFIG_TEMPLATE_FILE_PATH = "templates/init_template.yml"


def create_config_file(directory_path):
    import os
    import shutil
    template_yml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), CONFIG_TEMPLATE_FILE_PATH)
    result_yml_path = os.path.join(directory_path, CONFIG_FILENAME)
    shutil.copy(template_yml_path, result_yml_path)


def parse_yaml(path_to_yaml_file):
    import ruamel.yaml

    with open(path_to_yaml_file, 'r') as yaml_file:
        try:
            return ruamel.yaml.load(yaml_file, Loader=ruamel.yaml.Loader)
        except ruamel.yaml.YAMLError as exc:
            print(exc)


def get_option(option, config, alt=None, default=None):
    if alt is not None and alt != default:
        return alt

    if config and option in config:
        return config[option]

    return default


def calc_commit_impact(commit_sha, repo_path, results_count=10, commits_depth=50, check_all_commits=False, debug=False,
                       short_out=False):
    repo = Repo.init(repo_path)
    git_worker = GitRepoLoader(repo)
    ia = ImpactAnalyser()

    test_git_commit = repo.commit(commit_sha)
    test_item = GitUtils.commit_to_container(test_git_commit, repo)

    calc_type = NPNewCosineSimilarityCalculator

    all_items = git_worker.load_commits(test_git_commit, commits_depth=commits_depth if not check_all_commits else None)

    if not short_out:
        print(u"\033[1mOriginal item:\033[0m\n%s\n\t%s\n" % (test_item.name, test_item.message.replace('\n', '\n\t')))

    results = ia.impact_analysis_for_commit(test_item, all_items, results_count=results_count, debug=debug,
                                            calculator_type=calc_type)

    if not short_out:
        print(u"\033[1mResults:\033[0m")
    for container, distance in results:
        impact = 1.0 - distance
        if distance >= 1.0:
            continue
        if short_out:
            print(u'%s\t%f' % (container.name, impact))
        else:
            print(u'item \033[1m%s\033[0m (impact: \033[1m%f\033[0m)\n%s' % (container.name, impact, container.message.replace('\n', '\n\t')))


def main(app_options):
    yaml_config = None
    try:
        yaml_config = parse_yaml(app_options.yaml_config_file)
    except IOError as e:
        print "Configuration file not found. Run `gitimpact init` to create it"
        exit(1)

    result_options = {
        "commit_sha": get_option("commit", yaml_config, alt=app_options.commit),
        "repo_path": get_option("repo_path", yaml_config, alt=app_options.repo_path, default=os.getcwd()),
        "results_count": int(get_option("limit", yaml_config, alt=app_options.limit, default=-1)),
        "commits_depth": int(get_option("commits_depth", yaml_config, alt=app_options.commits_depth, default=-1)),
        "check_all_commits": bool(
            get_option("check_all_commits", yaml_config, alt=app_options.check_all_commits, default=False)),
        "debug": get_option("debug", yaml_config, alt=app_options.debug, default=False),
        "short_out": get_option("short_out", yaml_config, alt=app_options.short_out, default=False),
    }

    # print(result_options)

    calc_commit_impact(**result_options)
