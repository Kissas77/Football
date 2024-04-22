import gfootball.env as football_env
from config import BASEDIR
from agent.main import agent
from agent.opponent import agent as opp_agent


MAX_EPISODE_LEN = 5000
MAX_MATCH_LEN = 1

IS_OPP = False  # vs BT or AI
SAVE_VIDEO = False  # video
RENDER = False  # watch match

g_env_args = dict(
    env_name='11_vs_11_hard_stochastic',  # 11_vs_11_kaggle, 11_vs_11_hard_stochastic
    representation='raw',
    stacked=False,
    logdir=BASEDIR + '/videos/dumps',
    write_goal_dumps=True if SAVE_VIDEO else False,
    write_full_episode_dumps=True if SAVE_VIDEO else False,
    write_video=True if SAVE_VIDEO else False,
    render=RENDER,
    # number_of_right_players_agent_controls=1  # AI control right
)


def run():
    # init env
    global g_env_args
    env = football_env.create_environment(**g_env_args)
    # run
    for round_i in range(MAX_MATCH_LEN):
        env.reset()
        obs = env.observation()
        i = 0
        score = 0
        win = 0  # 0: equal, 1: self, 2: opp
        while True:
            obs = obs[0]
            act = agent(obs)
            # print(len(env.step(act)))
            obs, rew, done, _ = env.step(act)
            if i == MAX_EPISODE_LEN or done:
                break
            i += 1
            score += rew
        if score > 0:
            win = 1
        elif score < 0:
            win = 2
        print(f'round{round_i}, winner: {win}, score: {score}')
    env.close()


def run_opp():
    # init env
    global g_env_args
    g_env_args['env_name'] = '11_vs_11_kaggle'
    g_env_args['number_of_right_players_agent_controls'] = 1
    env = football_env.create_environment(**g_env_args)
    # run
    for round_i in range(MAX_MATCH_LEN):
        env.reset()
        obs, opp_obs = env.observation()
        score = 0
        win = 0  # 0: equal, 1: self, 2: opp
        i = 0
        while True:
            act = agent(obs)
            opp_act = opp_agent(opp_obs)
            act.extend(opp_act)
            obs_all, rew_all, done, _ = env.step(act)
            obs, opp_obs = obs_all
            if i == MAX_EPISODE_LEN or done:
                break
            i += 1
            score += rew_all[0]
        if score > 0:
            win = 1
        elif score < 0:
            win = 2
        print(f'round{round_i}, winner: {win}, score: {score}')
    env.close()


def main():
    run_opp() if IS_OPP else run()


if __name__ == '__main__':
    main()
