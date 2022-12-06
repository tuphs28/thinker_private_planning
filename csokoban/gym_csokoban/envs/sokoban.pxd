from libcpp cimport bool
from libcpp.string cimport string

cdef extern from "sokoban.cpp":
    pass

# Declare the class with cdef
cdef extern from "sokoban.h":

    cdef cppclass Sokoban:
        Sokoban() except +
        Sokoban(bool s, string level_dir, string img_dir) except +
        void reset(unsigned char* obs)
        void step(const int a, unsigned char* obs, float& reward, bool& done)
        int read_level(const int room_id)
        int print_level()
        void clone_state(unsigned char* room_status, int &step_n, bool &done) 
        void restore_state(const unsigned char* room_status, const int &step_n, const bool &done)
        int img_x, img_y, obs_x, obs_y, obs_n
        int step_n