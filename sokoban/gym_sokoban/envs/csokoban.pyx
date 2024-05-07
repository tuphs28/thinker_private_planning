# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# distutils: language = c++

from libcpp cimport bool
from libcpp.string cimport string
from cython.view cimport array as cvarray
import numpy as np
cimport numpy as np
np.import_array()
from .sokoban cimport Sokoban

cdef class cSokoban:
	cdef Sokoban c_sokoban
	cdef int obs_x, obs_y, obs_n, dim_x, dim_y, dim_z

	def __init__(self, bool small, string level_dir, string img_dir, int level_num, int dan_num, bool mini = True, int seed=0):
		self.c_sokoban = Sokoban(small, level_dir, img_dir, level_num, dan_num, seed, mini)	
		self.obs_x = self.c_sokoban.obs_x
		self.obs_y = self.c_sokoban.obs_y
		self.obs_n = self.c_sokoban.obs_n
		self.dim_x = self.obs_x - 2 if mini else self.obs_x
		self.dim_y = self.obs_y - 2 if mini else self.obs_y
		self.dim_z = 7 if mini else 3
		
	def reset(self):
		cdef np.ndarray obs = np.zeros((self.obs_n), dtype=np.dtype("u1"))
		cdef unsigned char[::1] obs_view = obs
		self.c_sokoban.reset(&obs_view[0])
		return obs.reshape((self.dim_x,self.dim_y,self.dim_z))

	def reset_level(self, int room_id):
		cdef np.ndarray obs = np.zeros((self.obs_n), dtype=np.dtype("u1"))
		cdef unsigned char[::1] obs_view = obs
		self.c_sokoban.reset_level(&obs_view[0], room_id)
		return obs.reshape((self.dim_x,self.dim_y,self.dim_z))		

	def step(self, int act):
		assert act >= 0 and act <= 4, "action should be in [0, 1, 2, 3, 4]"
		cdef np.ndarray obs = np.zeros((self.obs_n), dtype=np.dtype("u1"))
		cdef unsigned char[::1] obs_view = obs
		#obs = cvarray(shape=self.obs_n, itemsize=1, format="c")
		#cdef unsigned char[::1] obs_view = obs
		#cdef unsigned char obs[(shape=(self.obs_x-2) * (self.obs_y-2) * 7)]
		cdef float reward = 0.
		cdef bool done = False
		cdef bool truncated_done = False
		cdef bool cost = False
		self.c_sokoban.step(act, &obs_view[0], reward, done, truncated_done, cost)
		return obs.reshape((self.dim_x,self.dim_y,self.dim_z)), reward, done, {"step_n": self.step_n, "truncated_done": truncated_done, "cost": cost}

	def clone_state(self):
		cdef np.ndarray room_status = np.zeros(((self.obs_x-2) * (self.obs_y-2) * 7), dtype=np.dtype("u1"))
		cdef unsigned char[::1] room_status_view = room_status
		cdef int step_n = 0
		cdef bool done = False
		self.c_sokoban.clone_state(&room_status_view[0], step_n, done)
		return {"sokoban_room_status": room_status, 
		 		"sokoban_step_n": step_n,
		 		"sokoban_done": done}

	def restore_state(self, state):		
		cdef np.ndarray room_status = state["sokoban_room_status"]
		cdef int step_n  = state["sokoban_step_n"]
		cdef bool done = state["sokoban_done"]
		cdef unsigned char[::1] room_status_view = room_status
		self.c_sokoban.restore_state(&room_status_view[0], step_n, done)

	def seed(self, seed):		
		self.c_sokoban.set_seed(seed)

	@property
	def step_n(self):
		return self.c_sokoban.step_n

	@step_n.setter
	def step_n(self, step_n):
		self.c_sokoban.step_n = step_n		

	@property
	def obs_x(self):
		return self.c_sokoban.obs_x
		
	@property
	def obs_y(self):
		return self.c_sokoban.obs_y
"""
def main():
	env = cSokoban()
	obs = env.reset()
	plt.imshow(obs)
	plt.show()
	obs, reward, done, _ = env.step(2)
	plt.imshow(obs)
	plt.show()
	print(reward, done)
"""	