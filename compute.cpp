#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <fcntl.h>

extern "C" {
extern void compute(void);
}

int Iter = 1000;
int Rounds = 30;
using namespace std;
using namespace chrono;

int supress_stdout() {
  fflush(stdout);

  int ret = dup(1);
  int nullfd = open("/dev/null", O_WRONLY);
  // check nullfd for error omitted
  dup2(nullfd, 1);
  close(nullfd);

  return ret;
}

void resume_stdout(int fd) {
  fflush(stdout);
  dup2(fd, 1);
  close(fd);
}

double calculateSD(double data[]) {
  double sum = 0.0, mean, standardDeviation = 0.0;
  int i;

  for (i = 0; i < Rounds; ++i) {
    sum += data[i];
  }

  mean = sum / Rounds;

  for (i = 0; i < Rounds; ++i) {
    standardDeviation += pow(data[i] - mean, 2);
  }
  return sqrt(standardDeviation / Rounds);
}

double calculateMean(double data[]) {
  double sum = 0.0, mean = 0.0;
  int i;

  for (i = 0; i < Rounds; ++i) {
    sum += data[i];
  }

  mean = sum / Rounds;

  return mean;
}

void test() {
  double elaps[Rounds];
  int fd = supress_stdout();
  for (int k = 0; k < Rounds; k++) {

    high_resolution_clock::time_point iStart = high_resolution_clock::now();
    for (int i = 0; i < Iter; i++)
      compute();
    high_resolution_clock::time_point iFinished = high_resolution_clock::now();
    duration<double, std::milli> iElaps = iFinished - iStart;
    elaps[k] = iElaps.count();
  } 
  resume_stdout(fd);
  for (int k = 0; k < Rounds; k++) 
    cout << "round " << k << ": " << elaps[k] << endl;

  double sd = calculateSD(elaps);
  double mean = calculateMean(elaps);
  std::cout << "Binary "
            << " mean : " << mean << " sd " << sd << " .\n";
}

int main() {

  test();

  return 0;
}
