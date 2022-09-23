# Thompson sampling for the the multi-armed bandits (RL-course)

from http.client import NON_AUTHORITATIVE_INFORMATION
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # Data visualization library based on matplotlib
import scipy.stats as stats
import math
import time

# # set up matplotlib
# is_ipython = 'inline' in plt.get_backend()
# if is_ipython:
#     from IPython import display

np.random.seed(int(time.time())) # Numerical value that generates a new set or repeats pseudo-random numbers. The value in the numpy random seed saves the state of randomness.

# The probability of winning (exact value for each bandit), you can add more bandits here
Number_of_Bandits = 4
p_bandits = [0.5, 0.1, 0.8, 0.9] # Color: Blue, Orange, Green, Red


def bandit_run(index):
    if np.random.rand() >= p_bandits[index]: #random  probability to win or lose per machine
        return 0 # Lose
    else:
        return 1 # Win

def plot_steps(distribution, step, ax):
    plt.figure(1)
    plot_x = np.linspace(0.000, 1, 200) # create sequences of evenly spaced numbers structured as a NumPy array. # numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)
    ax.set_title(f'Step {step:d}')
    for d in distribution:
        y = d.pdf(plot_x)
        ax.plot(plot_x, y) # draw edges/curve of the plot
        ax.fill_between(plot_x, y, 0, alpha=0.1) # fill under the curve of the plot
    ax.set_ylim(bottom = 0) # limit plot axis

def plot_rewards(rewards_thompson, rewards_greedy_e, rewards_ucb):
    plt.figure(2)
    plt.title('Aveage Reward Comparision')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(rewards_thompson, color='green', label='Thompson')
    plt.plot(rewards_greedy_e, color='orange', label ='Greedy e')
    plt.plot(rewards_ucb, color = 'blue', label = 'UCB')
    plt.grid(axis='x', color='0.80')
    plt.legend(title='Parameter where:')
    plt.show()

def output_ratio(bandits_run_count_list, total_iterations):
    best_id = np.argmax(p_bandits)
    print("Number of optimal choices: ", bandits_run_count_list[best_id])
    print("Number of suboptimal choices: ", (total_iterations-bandits_run_count_list[best_id]))
    print(" ")


def get_epsilon(k):
    #easily adjustable epsilon decay TODO: make the factor 'a' dependent on the number of bandits
    a = 0.008
    e = 1 / math.exp(a*k)
    #e= 0.5/k
    return e

def greedy_epsilon(N):
    #N = 1000 # number of steps 
    bandit_run_count = [1] * Number_of_Bandits   # Array for Number of bandits try times, e.g. [0. 0, 0]
    bandit_win_count = [0] * Number_of_Bandits  # Array for Number of bandits win times, e.g. [0. 0, 0]
    bandit_value = [1] * Number_of_Bandits #Array for Number of Bandits with their expected value, init as 1 to favor unused ones at start

    random_runs = 0
    greedy_runs = 0     #Debugging

    #figure, ax = plt.subplots(4, 3, figsize=(9, 7)) # set the number of the plots in row and column and their sizes
    #ax = ax.flat # Iterator to plot

    average_reward = []
    for k in range(1, N):
    
        if np.random.rand() < get_epsilon(k):
            #Choose a random bandit if we are lower than our computed epsilon
            bandit_index = np.random.randint(1, Number_of_Bandits)
            success = bandit_run(bandit_index)
            bandit_run_count[bandit_index] += 1
            bandit_win_count[bandit_index] += success
            bandit_value[bandit_index] = bandit_win_count[bandit_index] / bandit_run_count[bandit_index]

            random_runs += 1


        else:
            #Choose the best bandit
            bandit_index = np.argmax(bandit_value)
            success = bandit_run(bandit_index)
            bandit_run_count[bandit_index] += 1
            bandit_win_count[bandit_index] += success
            bandit_value[bandit_index] = bandit_win_count[bandit_index] / bandit_run_count[bandit_index]

            greedy_runs += 1

        # Elemtwise division of lists using zip() and create new list [AvgRewardARM1, AvgRewardARM2, AvgRewardARM3, ...]
        # We do bandit_win_count[1]+bandit_win_count[2]+...) / (bandit_runing_count[0] + ...
        average_reward_list = ([n / m for n, m in zip(bandit_win_count, bandit_run_count)])

        # Get average of all bandits into only one reward value
        averaged_total_reward = 0
        for avged_arm_reward in (average_reward_list):
            averaged_total_reward += avged_arm_reward
        average_reward.append(averaged_total_reward)
    

    print("Greedy Epsilon Results:")

    print("Random runs: ", random_runs)
    print("Greedy runs: ", greedy_runs)
    output_ratio(bandit_run_count, N)
    
    
    #plt.tight_layout() # Adjust the padding between and around subplots.
    #plt.show()
    return average_reward
       

def ucb_sampler(N):
    #N = 1000 # number of steps for Thompson Sampling
    bandit_run_count = [1] * Number_of_Bandits   # Array for Number of bandits try times, e.g. [0. 0, 0]
    bandit_win_count = [0] * Number_of_Bandits  # Array for Number of bandits win times, e.g. [0. 0, 0]
    bandit_value = [1] * Number_of_Bandits # Array for Number of Bandits with their expected Q value, init as 1 to favor unused ones at start
    bandit_uncertainty = [1] * Number_of_Bandits # Array for Number of Bandits with ther uncetainty [U] value, init as 1 because we know nothing at start
    bandit_combined = [1] * Number_of_Bandits # Workaround so I can work with the np.argmax function, init 2 as q and u are initialized as 1

    average_reward = []
    for k in range(1, N):

        #Update all uncertainty values and the combined values accordingly
        for i in range(1, Number_of_Bandits):
            bandit_uncertainty[i] = math.sqrt((2*math.log(k)/bandit_run_count[i]))
            bandit_combined[i] = bandit_uncertainty[i] + bandit_value[i]

        #Choose the best bandit
        bandit_index = np.argmax(bandit_combined)
        success = bandit_run(bandit_index)
        bandit_run_count[bandit_index] += 1
        bandit_win_count[bandit_index] += success
        bandit_value[bandit_index] = bandit_win_count[bandit_index] / bandit_run_count[bandit_index]

        # Elemtwise division of lists using zip() and create new list [AvgRewardARM1, AvgRewardARM2, AvgRewardARM3, ...]
        # We do bandit_win_count[1]+bandit_win_count[2]+...) / (bandit_runing_count[0] + ...
        average_reward_list = ([n / m for n, m in zip(bandit_win_count, bandit_run_count)])

        # Get average of all bandits into only one reward value
        averaged_total_reward = 0
        for avged_arm_reward in (average_reward_list):
            averaged_total_reward += avged_arm_reward
        average_reward.append(averaged_total_reward)

        
    #plt.tight_layout() # Adjust the padding between and around subplots.
    #plt.show()
    print("UCB Results:")

    output_ratio(bandit_run_count, N)

    return average_reward

def thompson_sampler(N):
    #N = 1000 # number of steps for Thompson Sampling
    bandit_runing_count = [1] * Number_of_Bandits   # Array for Number of bandits try times, e.g. [0. 0, 0]
    bandit_win_count = [0] * Number_of_Bandits  # Array for Number of bandits win times, e.g. [0. 0, 0]

    #figure, ax = plt.subplots(4, 3, figsize=(9, 7)) # set the number of the plots in row and column and their sizes
    #ax = ax.flat # Iterator to plot

    average_reward = []
    for step in range(1, N):
        # Beta distribution and alfa beta calculation
        bandit_distribution = []
        for run_count, win_count in zip(bandit_runing_count, bandit_win_count): # create a tuple() of count and win
            bandit_distribution.append (stats.beta(a = win_count + 1, b = run_count - win_count + 1)) # calculate the main equation (beta distribution)

        prob_theta_samples = []
        # Theta probability sampeling for each bandit
        for p in bandit_distribution:
            prob_theta_samples.append(p.rvs(1)) #rvs method provides random samples of distibution

        # Select best bandit based on theta sample a bandit
        select_bandit = np.argmax(prob_theta_samples)

        # Run bandit and get both win count and run count
        bandit_win_count[select_bandit] += bandit_run(select_bandit) 
        bandit_runing_count[select_bandit] += 1

        #if step == 3 or step == 11 or (step % 100 == 1 and step <= 1000) :
        #    plot_steps(bandit_distribution, step - 1, next(ax))

        # Elemtwise division of lists using zip() and create new list [AvgRewardARM1, AvgRewardARM2, AvgRewardARM3, ...]
        # We do bandit_win_count[1]+bandit_win_count[2]+...) / (bandit_runing_count[0] + ...
        average_reward_list = ([n / m for n, m in zip(bandit_win_count, bandit_runing_count)])

        # Get average of all bandits into only one reward value
        averaged_total_reward = 0
        for avged_arm_reward in (average_reward_list):
            averaged_total_reward += avged_arm_reward
        average_reward.append(averaged_total_reward)

    print("Thompson Sampling Results:")
    output_ratio(bandit_runing_count, N)

    #plt.tight_layout() # Adjust the padding between and around subplots.
    #plt.show()

    #plot_rewards(average_reward)
    return average_reward

if __name__ == '__main__':
    N=1000 #Number of iterations
    ucb_reward = ucb_sampler(N)
    greedy_e_reward = greedy_epsilon(N)
    thompson_reward = thompson_sampler(N)
   
    plot_rewards(thompson_reward, greedy_e_reward, ucb_reward)