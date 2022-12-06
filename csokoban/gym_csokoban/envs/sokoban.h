#pragma once
#include <vector>
#include <string>

using namespace std;

enum class roomStatus : unsigned char { wall, empty, box_not_on_tar, box_on_tar, player_not_on_tar, player_on_tar, tar };
enum class action : unsigned char { noop, up, down, left, right };

constexpr char default_level_dir[] = "//home//sc//RS//thinker//csokoban//gym_csokoban//envs//boxoban-levels//unfiltered//train";
constexpr char default_img_dir[] = "//home//sc//RS//thinker//csokoban//gym_csokoban//envs//surface";

void read_bmp(const string &img_dir, const string &img_name, vector<unsigned char> &data);
char roomStatus_to_char(const roomStatus r);

class Sokoban {
public:
	Sokoban() : Sokoban(false) {};
	Sokoban(bool s) : Sokoban(false, default_level_dir, default_img_dir) {};
	Sokoban(bool s, string level_dir, string img_dir) :
		player_pos_x(0),
		player_pos_y(0),
		box_left(0),
		step_n(0),
		img_x(s ? small_img_x : large_img_x),
		img_y(s ? small_img_x : large_img_x),
		obs_x((s ? small_img_x : large_img_x)* room_x),
		obs_y((s ? small_img_x : large_img_x)* room_y),
		obs_n((s ? small_img_x * small_img_x : large_img_x * large_img_x)* room_x* room_y * 3),
		level_dir(level_dir),
		img_dir(img_dir),
		done(false),
		small(s),
		room_status(),
		spirites() {
		read_spirits();
	};
	static constexpr int room_x = 10, room_y = 10, small_img_x = 8, small_img_y = 8, large_img_x = 16, large_img_y = 16;
	void reset(unsigned char* obs);
	void step(const action a, unsigned char* obs, float& reward, bool& done);
	void step(const int a, unsigned char* obs, float& reward, bool& done);
	int read_level(const int room_id);
	int print_level();
	void clone_state(unsigned char* room_status, int& step_n, bool& done);
	void restore_state(const unsigned char* room_status, const int& step, const bool& done);
	int img_x, img_y, obs_x, obs_y, obs_n, step_n;
private:
	float move(const action a);
	void move_player(roomStatus& old_r, roomStatus& new_r);
	void move_pos(const action a, int& x, int& y);
	float move_box(roomStatus& old_r, roomStatus& new_r);
	void read_spirits();
	void render(unsigned char* obs);
	int player_pos_x, player_pos_y, box_left;
	bool done, small;
	string level_dir, img_dir;
	roomStatus room_status[room_y][room_x];
	vector<unsigned char> spirites[7];
};