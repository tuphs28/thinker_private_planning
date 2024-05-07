from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
from celluloid import Camera
from torch import swapaxes

def plot_mini_sokoban(state, legend=False):
    """Plot board state for mini sokoban

    Args:
        state (numpy.ndarray): 8x8x7 array representing board
        legend (bool, optional): if True, include legend explaining colours. Defaults to False.
    """
    if state.shape[0] != state.shape[1]:
        state = state.permute(1,2,0)
    mini_board = np.zeros(state.shape[:-1])
    for i in range(1,8):
        mini_board[(state[:,:,i-1] == 1)] = i
    cmap = colors.ListedColormap(['black', "white", "aqua", "gold", "green","magenta", "khaki"])
    bounds=[0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    board_img = plt.imshow(mini_board, interpolation='nearest', origin='lower', cmap=cmap, norm=norm)
    board_img.axes.get_yaxis().set_visible(False)
    board_img.axes.get_xaxis().set_visible(False)
    if legend:
        board_cbar = plt.colorbar(board_img, cmap=cmap, norm=norm, boundaries=bounds, ticks=bounds)
        board_cbar.ax.set_yticks(([1,2,3,4,5,6,7]))
        board_cbar.ax.tick_params(length=0)
        board_cbar_text = board_cbar.ax.set_yticklabels(["wall", "empty", "box", "box-on-target?", "player","player-on-target?", "target"])
    plt.show()


def make_gif_channels(states_list, layer, channels, gif_fps=0.5, gif_file="../viz/interesting_channels", gif_name="test", max_iter=30, max_activ=0, min_activ=0):
    """Make gif of multiple channels across timesteps

    Args:
        states_list (list): list of (agent_states, env_state) tuples, where agent_states is (batch,4,192) tensor of agent hidden states and env_state is (batch,x_dim,y_dim,z_dim) array of board state observations
        layer (int): layer to inspect (must be 0,1,2)
        channels (list): list of channels to inspect
        gif_fps (float, optional): gif speed. Defaults to 0.5.
        gif_file (str, optional): location to store gifs. Defaults to "../viz/interesting_channels".
        gif_name (str, optional): gif file name. Defaults to "test".
        max_iter (int, optional): max number of frames to see. Defaults to 30.
        max_activ (int, optional): cap activations above to this value for comparability if non-zero. Defaults to 0.
        min_activ (int, optional): cap activations below to this value for comparability if non-zero. Defaults to 0.
    """
    if max_activ and min_activ:
        for agent_states, env_state in states_list:
            agent_states[agent_states>max_activ] = max_activ
            agent_states[agent_states<min_activ] = min_activ

    fig, axs = plt.subplots(5,len(channels))
    for tick_idx in range(5):
        for channel_idx in range(len(channels)):
            axs[tick_idx, channel_idx].axis("off")
    camera = Camera(fig)
    i = 0
    for agent_states, env_state in states_list:
        env_state = swapaxes(swapaxes(env_state[-3:].detach().cpu(), 0, 2), 0, 1).numpy()
        for channel_idx, channel in enumerate(channels):
            axs[0,channel_idx].set_title(channel)
            axs[0, channel_idx].imshow(env_state, interpolation="nearest")
            for tick_idx in range(4):
                if max_activ and min_activ:
                    axs[tick_idx+1, channel_idx].imshow(agent_states[tick_idx,layer,channel,:,:].detach(), vmin=min_activ, vmax=max_activ)
                else:
                    axs[tick_idx+1, channel_idx].imshow(agent_states[tick_idx,layer,channel,:,:].detach())
        for layer_idx in range(5):
            for tick_idx in range(len(channels)):
                axs[layer_idx, tick_idx].axis("off")
        plt.pause(0.1)
        camera.snap()
        i += 1
        if i == max_iter:
            break
    animation = camera.animate()
    animation.save(gif_file+"/"+f'{gif_name}.gif', writer='PillowWriter', fps=0.5) 


def make_gif_channel_across_envs(states_list, layer, channel, gif_fps=0.5, gif_name="test", gif_file="../viz/channel_vizs", max_iter=30):
    """Make gif of single channels for multiple environments; states_list needs to be a list where each entry is a tuple of (agent_states, env_state) entries across envs

    Args:
        states_list (list):  list where each entry is a tuple of (agent_states, env_state) entries across envs
        layer (int): layer to inspect (must be 0,1,2)
        channels (int): channels to inspect
        gif_fps (float, optional): gif speed. Defaults to 0.5.
        gif_file (str, optional): location to store gifs. Defaults to "../viz/interesting_channels".
        gif_name (str, optional): gif file name. Defaults to "test".
        max_iter (int, optional): max number of frames to see. Defaults to 30.
    """
    fig, axs = plt.subplots(5,len(states_list[0]))
    fig.suptitle(f"Layer {layer}, channel {channel}")
    for tick_idx in range(5):
        for env_idx in range(len(states_list[0])):
            axs[tick_idx, env_idx].axis("off")
    camera = Camera(fig)
    for i in range(max_iter):
        for env_idx, (agent_states, env_state) in enumerate(states_list[i]):
            env_state = swapaxes(swapaxes(env_state[-3:].detach().cpu(), 0, 2), 0, 1).numpy()
            axs[0, env_idx].imshow(env_state, interpolation="nearest")
            for tick_idx in range(4):
                axs[tick_idx+1, env_idx].imshow(agent_states[tick_idx,layer,channel,:,:].detach())
        plt.pause(0.1)
        camera.snap()
    animation = camera.animate()
    animation.save(gif_file+"/"+f'{gif_name}.gif', writer='PillowWriter', fps=gif_fps)   