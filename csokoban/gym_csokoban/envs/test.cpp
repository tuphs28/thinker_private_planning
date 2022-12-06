#include "sokoban.h"
#include <iostream>
using namespace std;

int main()
{
	cout << "size of sokoban: " << sizeof(Sokoban) << endl;
	Sokoban sokoban;
	unsigned char* obs = new unsigned char[sokoban.obs_n];
	float reward = 0.;
	bool done = false;
	sokoban.reset(obs);
	//sokoban.read_level(1002);
	sokoban.print_level();
	int a;
	while (cin >> a) {
		sokoban.step(a, obs, reward, done);
		sokoban.print_level();
		cout << "reward: " << reward << " done: " << done << endl;
	}
	delete[] obs;
}
