#pragma once
#include <string>
#include <iostream>
using namespace std;
class ArgumentParser {
private:

public:
    int argc;
    char **argv;

    bool canHaveSource;

    bool hasInput;
    bool hasSourceNode;
    string input;
    float adviseK = 0.0f;
    int sourceNode;
    int method;
    string algo;
    int model;
    int testTimes;
    ArgumentParser(int argc, char **argv, bool canHaveSource);

    bool Parse();

    string GenerateHelpString();

};