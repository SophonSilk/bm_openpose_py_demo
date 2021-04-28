#include <stdio.h>
#include <signal.h>
#include <unistd.h>


#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <sys/time.h>
#include <sys/stat.h>
#include <time.h>
#include <iostream>
#include <algorithm>
#include <stdio.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <fstream>
#include "bmruntime_interface.h"
#include <dirent.h>
#include "boost/make_shared.hpp"

using namespace std;
using namespace cv;


namespace op
{

    static  auto  timerPast=std::chrono::high_resolution_clock::now();

    void stopWatch(string title = "") {
        if (title != "") {
            auto timerNow = std::chrono::high_resolution_clock::now();
            const auto totalTimeSec =
                    (double) std::chrono::duration_cast<std::chrono::nanoseconds>(timerNow - timerPast).count() * 1e-9;
            std::cout << title << " (s) : " << (totalTimeSec) << std::endl;
            timerPast = std::chrono::high_resolution_clock::now();
        } else {
            timerPast = std::chrono::high_resolution_clock::now();
        }
    }

    float caldis(cv::Point p1, cv::Point p2)
    {
        return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
    }

    template<typename T>
    inline T fastMax(const T a, const T b)
    {
        return (a > b ? a : b);
    }
    template<typename T>
    inline T fastMin(const T a, const T b)
    {
        return (a < b ? a : b);
    }
    template<typename T>
    inline int intRound(const T a)
    {
        return int(a+0.5f);
    }

#define POSE_MAX_PEOPLE  96   //the max people num in the image which will be finded


    std::vector<unsigned int> getPoseMapIndex()
    {
        return  std::vector<unsigned int>{
                12,13, 20,21, 14,15, 16,17, 22,23, 24,25, 0,1, 2,3, 4,5, 6,7, 8,9, 10,11, 28,29, 30,31, 34,35, 32,33, 36,37, 18,19, 26,27
        };
    }
    std::vector<int> getPosePartPairsStar()
    {
        return std::vector<int>{};
    }
    std::vector<unsigned int> getPosePartPairs()
    {
        return  std::vector<unsigned int>{
                1,2,   1,5,   2,3,   3,4,   5,6,   6,7,   1,8,   8,9,   9,10,  1,11,  11,12, 12,13,  1,0,   0,14, 14,16,  0,15, 15,17,  2,16,  5,17
        };
    }

    struct DII
    {
        double a;
        int b;
        int c;

    };
    struct FII
    {
        float a;
        int b;
        int c;

    };
    struct IID
    {
        int a;
        int b;
        double c;
    };
    bool greaterdii(const DII & m1, const DII & m2) {
        return m1.a > m2.a;
    }



    template <typename T>
    inline T getScoreAB(const int i, const int j, const T* const candidateAPtr, const T* const candidateBPtr,
                        const T* const mapX, const T* const mapY, const cv::Point& heatMapSize,
                        const T interThreshold, const T interMinAboveThreshold)
    {
        try
        {
            const auto vectorAToBX = candidateBPtr[j*3] - candidateAPtr[i*3];
            const auto vectorAToBY = candidateBPtr[j*3+1] - candidateAPtr[i*3+1];
            const auto vectorAToBMax = fastMax(std::abs(vectorAToBX), std::abs(vectorAToBY));
            const auto numberPointsInLine = fastMax(
                5, fastMin(25, intRound(std::sqrt(5*vectorAToBMax))));
            const auto vectorNorm = T(std::sqrt( vectorAToBX*vectorAToBX + vectorAToBY*vectorAToBY ));
            // If the peaksPtr are coincident. Don't connect them.
            if (vectorNorm > 1e-6)
            {
                const auto sX = candidateAPtr[i*3];
                const auto sY = candidateAPtr[i*3+1];
                const auto vectorAToBNormX = vectorAToBX/vectorNorm;
                const auto vectorAToBNormY = vectorAToBY/vectorNorm;

                auto sum = T(0);
                auto count = 0u;
                const auto vectorAToBXInLine = vectorAToBX/numberPointsInLine;
                const auto vectorAToBYInLine = vectorAToBY/numberPointsInLine;
                for (auto lm = 0; lm < numberPointsInLine; lm++)
                {
                    const auto mX = fastMax(
                        0, fastMin(heatMapSize.x-1, intRound(sX + lm*vectorAToBXInLine)));
                    const auto mY = fastMax(
                        0, fastMin(heatMapSize.y-1, intRound(sY + lm*vectorAToBYInLine)));
                    const auto idx = mY * heatMapSize.x + mX;
                    const auto score = (vectorAToBNormX*mapX[idx] + vectorAToBNormY*mapY[idx]);
                    if (score > interThreshold)
                    {
                        sum += score;
                        count++;
                    }
                }
                if (count/(float)numberPointsInLine > interMinAboveThreshold)
                    return sum/count;
            }
            return T(0);
        }
        catch (const std::exception& e)
        {
            //error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return T(0);
        }
    }
    template <typename T>
    inline T getScore0B(const int bodyPart0, const T* const candidate0Ptr, const int i, const int j,
                        const int bodyPartA, const int bodyPartB, const T* const candidateBPtr,
                        const T* const heatMapPtr, const cv::Point& heatMapSize,
                        const T interThreshold, const T interMinAboveThreshold, const int peaksOffset,
                        const int heatMapOffset, const int numberBodyPartsAndBkg,
                        const std::vector<std::pair<std::vector<int>, double>>& subsets,
                        const std::vector<int>& bodyPartPairsStar)
    {
        try
        {
            // A is already in the subsets, find its connection B
            const auto pairIndex2 = bodyPartPairsStar[bodyPartB];
            const auto* mapX0 = heatMapPtr + (numberBodyPartsAndBkg + pairIndex2) * heatMapOffset;
            const auto* mapY0 = heatMapPtr + (numberBodyPartsAndBkg + pairIndex2+1) * heatMapOffset;
            const int indexA = bodyPartA*peaksOffset + i*3 + 2;
            for (auto& subset : subsets)
            {
                const auto index0 = subset.first[bodyPart0];
                if (index0 > 0)
                {
                    // Found partA in a subsets, add partB to same one.
                    if (subset.first[bodyPartA] == indexA)
                    {
                        // index0 = std::get<0>(abConnection) = bodyPart0*peaksOffset + i0*3 + 2
                        // i0 = (index0 - 2 - bodyPart0*peaksOffset)/3
                        const auto i0 = (index0 - 2 - bodyPart0*peaksOffset)/3.;
                        return getScoreAB(i0, j, candidate0Ptr, candidateBPtr, mapX0, mapY0,
                                          heatMapSize, interThreshold, interMinAboveThreshold);
                    }
                }
            }
            return T(0);
        }
        catch (const std::exception& e)
        {
            //error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return T(0);
        }
    }

    template <typename T>
    std::vector<std::pair<std::vector<int>, double>> generateInitialSubsets(
        const T* const heatMapPtr, const T* const peaksPtr, const cv::Point& heatMapSize,
        const int maxPeaks, const T interThreshold, const T interMinAboveThreshold,
        const std::vector<unsigned int>& bodyPartPairs, const unsigned int numberBodyParts,
        const unsigned int numberBodyPartPairs, const unsigned int subsetCounterIndex)
    {
        try
        {
            // std::vector<std::pair<std::vector<int>, double>> refers to:
            //     - std::vector<int>: [body parts locations, #body parts found]
            //     - double: subset score
            std::vector<std::pair<std::vector<int>, double>> subsets;
            const auto& mapIdx = getPoseMapIndex();
            const auto numberBodyPartsAndBkg = numberBodyParts + 1;
            const auto subsetSize = numberBodyParts+1;
            const auto peaksOffset = 3*(maxPeaks+1);
            const auto heatMapOffset = heatMapSize.x * heatMapSize.y;
            const auto& bodyPartPairsStar = getPosePartPairsStar();
            // Star-PAF
            const auto bodyPart0 = 1;
            const auto* candidate0Ptr = peaksPtr + bodyPart0*peaksOffset;
            const auto number0 = intRound(candidate0Ptr[0]);
            // Iterate over it PAF connection, e.g. neck-nose, neck-Lshoulder, etc.
            for (auto pairIndex = 0u; pairIndex < numberBodyPartPairs; pairIndex++)
            {
                const auto bodyPartA = bodyPartPairs[2*pairIndex];
                const auto bodyPartB = bodyPartPairs[2*pairIndex+1];
                const auto* candidateAPtr = peaksPtr + bodyPartA*peaksOffset;
                const auto* candidateBPtr = peaksPtr + bodyPartB*peaksOffset;
                const auto numberA = intRound(candidateAPtr[0]);
                const auto numberB = intRound(candidateBPtr[0]);

                // E.g. neck-nose connection. If one of them is empty (e.g. no noses detected)
                // Add the non-empty elements into the subsets
                if (numberA == 0 || numberB == 0)
                {
                    // E.g. neck-nose connection. If no necks, add all noses
                    // Change w.r.t. other
                    if (numberA == 0) // numberB == 0 or not
                    {
                        for (auto i = 1; i <= numberB; i++)
                        {
                            bool found = false;
                            for (const auto& subset : subsets)
                            {
                                const auto off = (int)bodyPartB*peaksOffset + i*3 + 2;
                                if (subset.first[bodyPartB] == off)
                                {
                                    found = true;
                                    break;
                                }
                            }
                            if (!found)
                            {
                                // Refinement: star-PAF
                                // Look for root-B connection
                                auto maxScore = T(0);
                                auto maxScoreIndex = -1;
                                // Star-PAF --> found
                                if (maxScore > 0)
                                {
                                    // bool found = false;
                                    for (auto& subset : subsets)
                                    {
                                        const int index0 = bodyPart0*peaksOffset + maxScoreIndex*3 + 2;
                                        // Found partA in a subsets, add partB to same one.
                                        if (subset.first[bodyPart0] == index0)
                                        {
                                            const auto indexB = bodyPartB*peaksOffset + i*3 + 2;
                                            subset.first[bodyPartB] = indexB;
                                            subset.first[subsetCounterIndex]++;
                                            subset.second += peaksPtr[indexB] + maxScore;
                                            // found = true;
                                            break;
                                        }
                                    }
                                }
                                // Add new subset with this element - Non-star-PAF code or Star-PAF when not found
                                else // if (!found)
                                {
                                    std::vector<int> rowVector(subsetSize, 0);
                                    // Store the index
                                    rowVector[ bodyPartB ] = bodyPartB*peaksOffset + i*3 + 2;
                                    // Last number in each row is the parts number of that person
                                    rowVector[subsetCounterIndex] = 1;
                                    const auto subsetScore = candidateBPtr[i*3+2];
                                    // Second last number in each row is the total score
                                    subsets.emplace_back(std::make_pair(rowVector, subsetScore));
                                }
                            }
                        }
                    }
                    // E.g. neck-nose connection. If no noses, add all necks
                    else // if (numberA != 0 && numberB == 0)
                    {
                        for (auto i = 1; i <= numberA; i++)
                        {
                            bool found = false;
                            const auto indexA = bodyPartA;
                            for (const auto& subset : subsets)
                            {
                                const auto off = (int)bodyPartA*peaksOffset + i*3 + 2;
                                if (subset.first[indexA] == off)
                                {
                                    found = true;
                                    break;
                                }
                            }
                            if (!found)
                            {
                                std::vector<int> rowVector(subsetSize, 0);
                                // Store the index
                                rowVector[ bodyPartA ] = bodyPartA*peaksOffset + i*3 + 2;
                                // Last number in each row is the parts number of that person
                                rowVector[subsetCounterIndex] = 1;
                                // Second last number in each row is the total score
                                const auto subsetScore = candidateAPtr[i*3+2];
                                subsets.emplace_back(std::make_pair(rowVector, subsetScore));
                            }
                        }
                    }
                }
                // E.g. neck-nose connection. If necks and noses, look for maximums
                else // if (numberA != 0 && numberB != 0)
                {
                    // (score, x, y). Inverted order for easy std::sort
                    std::vector<std::tuple<double, int, int>> allABConnections;
                    // Note: Problem of this function, if no right PAF between A and B, both elements are discarded.
                    // However, they should be added indepently, not discarded
                    {
                        const auto* mapX = heatMapPtr + (numberBodyPartsAndBkg + mapIdx[2*pairIndex]) * heatMapOffset;
                        const auto* mapY = heatMapPtr + (numberBodyPartsAndBkg + mapIdx[2*pairIndex+1]) * heatMapOffset;
                        // E.g. neck-nose connection. For each neck
                        for (auto i = 1; i <= numberA; i++)
                        {
                            // E.g. neck-nose connection. For each nose
                            for (auto j = 1; j <= numberB; j++)
                            {
                                // Initial PAF
                                auto scoreAB = getScoreAB(i, j, candidateAPtr, candidateBPtr, mapX, mapY,
                                                          heatMapSize, interThreshold, interMinAboveThreshold);

                                // E.g. neck-nose connection. If possible PAF between neck i, nose j --> add
                                // parts score + connection score
                                if (scoreAB > 1e-6)
                                    allABConnections.emplace_back(std::make_tuple(scoreAB, i, j));
                            }
                        }
                    }

                    // select the top minAB connection, assuming that each part occur only once
                    // sort rows in descending order based on parts + connection score
                    if (!allABConnections.empty())
                        std::sort(allABConnections.begin(), allABConnections.end(),
                                  std::greater<std::tuple<double, int, int>>());

                    std::vector<std::tuple<int, int, double>> abConnections; // (x, y, score)
                    {
                        const auto minAB = fastMin(numberA, numberB);
                        std::vector<int> occurA(numberA, 0);
                        std::vector<int> occurB(numberB, 0);
                        auto counter = 0;
                        for (auto row = 0u; row < allABConnections.size(); row++)
                        {
                            const auto score = std::get<0>(allABConnections[row]);
                            const auto i = std::get<1>(allABConnections[row]);
                            const auto j = std::get<2>(allABConnections[row]);
                            if (!occurA[i-1] && !occurB[j-1])
                            {
                                abConnections.emplace_back(std::make_tuple(bodyPartA*peaksOffset + i*3 + 2,
                                                                           bodyPartB*peaksOffset + j*3 + 2,
                                                                           score));
                                counter++;
                                if (counter==minAB)
                                    break;
                                occurA[i-1] = 1;
                                occurB[j-1] = 1;
                            }
                        }
                    }

                    // Cluster all the body part candidates into subsets based on the part connection
                    if (!abConnections.empty())
                    {
                        // initialize first body part connection 15&16
                        if (pairIndex==0)
                        {
                            for (const auto& abConnection : abConnections)
                            {
                                std::vector<int> rowVector(numberBodyParts+3, 0);
                                const auto indexA = std::get<0>(abConnection);
                                const auto indexB = std::get<1>(abConnection);
                                const auto score = std::get<2>(abConnection);
                                rowVector[bodyPartPairs[0]] = indexA;
                                rowVector[bodyPartPairs[1]] = indexB;
                                rowVector[subsetCounterIndex] = 2;
                                // add the score of parts and the connection
                                const auto subsetScore = peaksPtr[indexA] + peaksPtr[indexB] + score;
                                subsets.emplace_back(std::make_pair(rowVector, subsetScore));
                            }
                        }
                        // Add ears connections (in case person is looking to opposite direction to camera)
                        // Note: This has some issues:
                        //     - It does not prevent repeating the same keypoint in different people
                        //     - Assuming I have nose,eye,ear as 1 subset, and whole arm as another one, it will not
                        //       merge them both
                        else if ((numberBodyParts == 18 && (pairIndex==17 || pairIndex==18)))
                        {
                            for (const auto& abConnection : abConnections)
                            {
                                const auto indexA = std::get<0>(abConnection);
                                const auto indexB = std::get<1>(abConnection);
                                for (auto& subset : subsets)
                                {
                                    auto& subsetA = subset.first[bodyPartA];
                                    auto& subsetB = subset.first[bodyPartB];
                                    if (subsetA == indexA && subsetB == 0)
                                    {
                                        subsetB = indexB;
                                        // // This seems to harm acc 0.1% for BODY_25
                                        // subset.first[subsetCounterIndex]++;
                                    }
                                    else if (subsetB == indexB && subsetA == 0)
                                    {
                                        subsetA = indexA;
                                        // // This seems to harm acc 0.1% for BODY_25
                                        // subset.first[subsetCounterIndex]++;
                                    }
                                }
                            }
                        }
                        else
                        {
                            // A is already in the subsets, find its connection B
                            for (const auto& abConnection : abConnections)
                            {
                                const auto indexA = std::get<0>(abConnection);
                                const auto indexB = std::get<1>(abConnection);
                                const auto score = std::get<2>(abConnection);
                                bool found = false;
                                for (auto& subset : subsets)
                                {
                                    // Found partA in a subsets, add partB to same one.
                                    if (subset.first[bodyPartA] == indexA)
                                    {
                                        subset.first[bodyPartB] = indexB;
                                        subset.first[subsetCounterIndex]++;
                                        subset.second += peaksPtr[indexB] + score;
                                        found = true;
                                        break;
                                    }
                                }
                                // Not found partA in subsets, add new subsets element
                                if (!found)
                                {
                                    std::vector<int> rowVector(subsetSize, 0);
                                    rowVector[bodyPartA] = indexA;
                                    rowVector[bodyPartB] = indexB;
                                    rowVector[subsetCounterIndex] = 2;
                                    const auto subsetScore = peaksPtr[indexA] + peaksPtr[indexB] + score;
                                    subsets.emplace_back(std::make_pair(rowVector, subsetScore));
                                }
                            }
                        }
                    }
                }
            }
            return subsets;
        }
        catch (const std::exception& e)
        {
            //error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }

    template <typename T>
    void removeSubsetsBelowThresholds(std::vector<int>& validSubsetIndexes, int& numberPeople,
                                      const std::vector<std::pair<std::vector<int>, double>>& subsets,
                                      const unsigned int subsetCounterIndex, const unsigned int numberBodyParts,
                                      const int minSubsetCnt, const T minSubsetScore)
    {
        try
        {
            // Delete people below the following thresholds:
                // a) minSubsetCnt: removed if less than minSubsetCnt body parts
                // b) minSubsetScore: removed if global score smaller than this
                // c) POSE_MAX_PEOPLE: keep first POSE_MAX_PEOPLE people above thresholds
            numberPeople = 0;
            validSubsetIndexes.clear();
            validSubsetIndexes.reserve(fastMin((size_t)POSE_MAX_PEOPLE, subsets.size()));
            for (auto index = 0u ; index < subsets.size() ; index++)
            {
                auto subsetCounter = subsets[index].first[subsetCounterIndex];
                // Foot keypoints do not affect subsetCounter (too many false positives,
                // same foot usually appears as both left and right keypoints)
                // Pros: Removed tons of false positives
                // Cons: Standalone leg will never be recorded
                const auto subsetScore = subsets[index].second;
                if (subsetCounter >= minSubsetCnt && (subsetScore/subsetCounter) >= minSubsetScore)
                {
                    numberPeople++;
                    validSubsetIndexes.emplace_back(index);
                    if (numberPeople == POSE_MAX_PEOPLE)
                        break;
                }
                else if ((subsetCounter < 1 && numberBodyParts != 25) || subsetCounter < 0)
                    ;
                    //error("Bad subsetCounter (" + std::to_string(subsetCounter) + "). Bug in this" " function if this happens.", __LINE__, __FUNCTION__, __FILE__);
            }
        }
        catch (const std::exception& e)
        {
            //error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template <typename T>
    void subsetsToPoseKeypointsAndScores(std::vector<float> * poseKeypoints, std::vector<float> * poseScores, const T scaleFactor,
                                         const std::vector<std::pair<std::vector<int>, double>>& subsets,
                                         const std::vector<int>& validSubsetIndexes, const T* const peaksPtr,
                                         const int numberPeople, const unsigned int numberBodyParts,
                                         const unsigned int numberBodyPartPairs)
    {
        try
        {
            if (numberPeople > 0)
            {
                // Initialized to 0 for non-found keypoints in people
                (*poseKeypoints)=std::vector<float>((int)(numberPeople*numberBodyParts*3));
                (*poseScores)=std::vector<float>(numberPeople);
            }
            else
            {
                (*poseKeypoints)=std::vector<float>();
                (*poseScores)=std::vector<float>();
            }
            const auto numberBodyPartsAndPAFs = numberBodyParts + numberBodyPartPairs;
            for (auto person = 0u ; person < validSubsetIndexes.size() ; person++)
            {
                const auto& subsetPair = subsets[validSubsetIndexes[person]];
                const auto& subset = subsetPair.first;
                for (auto bodyPart = 0u; bodyPart < numberBodyParts; bodyPart++)
                {
                    const auto baseOffset = (person*numberBodyParts + bodyPart) * 3;
                    const auto bodyPartIndex = subset[bodyPart];
                    if (bodyPartIndex > 0)
                    {
                        (*poseKeypoints)[baseOffset] = peaksPtr[bodyPartIndex-2] * scaleFactor+0.5;
                        (*poseKeypoints)[baseOffset + 1] = peaksPtr[bodyPartIndex-1] * scaleFactor+0.5;
                        (*poseKeypoints)[baseOffset + 2] = peaksPtr[bodyPartIndex];
                    }
                }
                (*poseScores)[person] = subsetPair.second / (float)(numberBodyPartsAndPAFs);
            }
        }
        catch (const std::exception& e)
        {
            //error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void connectBodyPartsCpu(std::vector<float> * poseKeypoints, std::vector<float> * poseScores, const float* const heatMapPtr,
                             const float* const peaksPtr, const cv::Point& heatMapSize,
                             const int maxPeaks, const float interMinAboveThreshold, const float interThreshold,
                             const int minSubsetCnt, const float minSubsetScore, const float scaleFactor)
    {
        try
        {
            // Parts Connection
            const auto& bodyPartPairs = getPosePartPairs();
            const auto numberBodyParts = 18;
            const auto numberBodyPartPairs = bodyPartPairs.size() / 2;
            const auto subsetCounterIndex = numberBodyParts;
            //if (numberBodyParts == 0)
                //error("Invalid value of numberBodyParts, it must be positive, not " + std::to_string(numberBodyParts),
                      //__LINE__, __FUNCTION__, __FILE__);

            // std::vector<std::pair<std::vector<int>, double>> refers to:
            //     - std::vector<int>: [body parts locations, #body parts found]
            //     - double: subset score
            const auto subsets = generateInitialSubsets(
                heatMapPtr, peaksPtr, heatMapSize, maxPeaks, interThreshold, interMinAboveThreshold,
                bodyPartPairs, numberBodyParts, numberBodyPartPairs, subsetCounterIndex);

            // Delete people below the following thresholds:
                // a) minSubsetCnt: removed if less than minSubsetCnt body parts
                // b) minSubsetScore: removed if global score smaller than this
                // c) POSE_MAX_PEOPLE: keep first POSE_MAX_PEOPLE people above thresholds
            int numberPeople;
            std::vector<int> validSubsetIndexes;
            validSubsetIndexes.reserve(fastMin((size_t)POSE_MAX_PEOPLE, subsets.size()));
            removeSubsetsBelowThresholds(validSubsetIndexes, numberPeople, subsets, subsetCounterIndex,
                                         numberBodyParts, minSubsetCnt, minSubsetScore);

            // Fill and return poseKeypoints
            subsetsToPoseKeypointsAndScores(poseKeypoints, poseScores, scaleFactor, subsets, validSubsetIndexes,
                                            peaksPtr, numberPeople, numberBodyParts, numberBodyPartPairs);
        }
        catch (const std::exception& e)
        {
            //error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }


//    template void connectBodyPartsCpu(std::vector<float> * poseKeypoints, std::vector<float> * poseScores,
//                                      const float* const heatMapPtr, const float* const peaksPtr,
//                                      const cv::Point& heatMapSize,
//                                      const int maxPeaks, const float interMinAboveThreshold,
//                                      const float interThreshold, const int minSubsetCnt,
//                                      const float minSubsetScore, const float scaleFactor);
//
//    template void connectBodyPartsCpu(std::vector<float> * poseKeypoints, std::vector<float> * poseScores,
//                                      const double* const heatMapPtr, const double* const peaksPtr,
//                                      const cv::Point& heatMapSize,
//                                      const int maxPeaks, const double interMinAboveThreshold,
//                                      const double interThreshold, const int minSubsetCnt,
//                                      const double minSubsetScore, const double scaleFactor);
}

typedef std::vector<float> Prediction;


template <typename T>
void nmsRegisterKernelCPU(int* kernelPtr, const T* const sourcePtr, const int w, const int h,
                          const T& threshold, const int x, const int y);
template <typename T>
void nmsAccuratePeakPosition(const T* const sourcePtr, const int& peakLocX, const int& peakLocY,
                             const int& width, const int& height, T* output);
void nmsCpu(float * targetPtr, int* kernelPtr, const float * const sourcePtr, const float threshold,
            const std::array<int, 4>& targetSize, const std::array<int, 4>& sourceSize);



template <typename T>
void nmsRegisterKernelCPU(int* kernelPtr, const T* const sourcePtr, const int w, const int h,
                          const T& threshold, const int x, const int y)
{
    // We have three scenarios for NMS, one for the border, 1 for the 1st inner border, and
    // 1 for the rest. cv::resize adds artifacts around the 1st inner border, causing two
    // maximas to occur side by side. Eg. [1 1 0.8 0.8 0.5 ..]. The CUDA kernel gives
    // [0.8 1 0.8 0.8 0.5 ..] Hence for this special case in the 1st inner border, we look at the
    // visible regions.

    const auto index = y*w + x;

    //std::cout<<"source data:"<<sourcePtr[index]<<std::endl;

    if (1 < x && x < (w-2) && 1 < y && y < (h-2))
    {
        const auto value = sourcePtr[index];
        if (value > threshold)
        {
            const auto topLeft     = sourcePtr[(y-1)*w + x-1];
            const auto top         = sourcePtr[(y-1)*w + x];
            const auto topRight    = sourcePtr[(y-1)*w + x+1];
            const auto left        = sourcePtr[    y*w + x-1];
            const auto right       = sourcePtr[    y*w + x+1];
            const auto bottomLeft  = sourcePtr[(y+1)*w + x-1];
            const auto bottom      = sourcePtr[(y+1)*w + x];
            const auto bottomRight = sourcePtr[(y+1)*w + x+1];

            if (value > topLeft && value > top && value > topRight
                && value > left && value > right
                && value > bottomLeft && value > bottom && value > bottomRight)
                kernelPtr[index] = 1;
            else
                kernelPtr[index] = 0;
        }
        else
            kernelPtr[index] = 0;
    }
    else if (x == 1 || x == (w-2) || y == 1 || y == (h-2))
    {
        //kernelPtr[index] = 0;
        const auto value = sourcePtr[index];
        if (value > threshold)
        {
            const auto topLeft      = ((0 < x && 0 < y)         ? sourcePtr[(y-1)*w + x-1]  : threshold);
            const auto top          = (0 < y                    ? sourcePtr[(y-1)*w + x]    : threshold);
            const auto topRight     = ((0 < y && x < (w-1))     ? sourcePtr[(y-1)*w + x+1]  : threshold);
            const auto left         = (0 < x                    ? sourcePtr[    y*w + x-1]  : threshold);
            const auto right        = (x < (w-1)                ? sourcePtr[y*w + x+1]      : threshold);
            const auto bottomLeft   = ((y < (h-1) && 0 < x)     ? sourcePtr[(y+1)*w + x-1]  : threshold);
            const auto bottom       = (y < (h-1)                ? sourcePtr[(y+1)*w + x]    : threshold);
            const auto bottomRight  = ((x < (w-1) && y < (h-1)) ? sourcePtr[(y+1)*w + x+1]  : threshold);

            if (value >= topLeft && value >= top && value >= topRight
                && value >= left && value >= right
                && value >= bottomLeft && value >= bottom && value >= bottomRight)
                kernelPtr[index] = 1;
            else
                kernelPtr[index] = 0;
        }
        else
            kernelPtr[index] = 0;
    }
    else
        kernelPtr[index] = 0;
}
template <typename T>
void nmsAccuratePeakPosition(const T* const sourcePtr, const int& peakLocX, const int& peakLocY,
                             const int& width, const int& height, T* output)
{
    //std::cout<<"point "<<std::endl;
    //std::cout<<peakLocX<<std::endl;
    //std::cout<<peakLocY<<std::endl;
    T xAcc = 0.f;
    T yAcc = 0.f;
    T scoreAcc = 0.f;
    const auto dWidth = 3;
    const auto dHeight = 3;
    for (auto dy = -dHeight ; dy <= dHeight ; dy++)
    {
        const auto y = peakLocY + dy;
        if (0 <= y && y < height) // Default height = 368
        {
            for (auto dx = -dWidth ; dx <= dWidth ; dx++)
            {
                const auto x = peakLocX + dx;
                if (0 <= x && x < width) // Default width = 656
                {
                    const auto score = sourcePtr[y * width + x];
                    if (score > 0)
                    {
                        xAcc += x*score;
                        yAcc += y*score;
                        scoreAcc += score;
                    }
                }
            }
        }
    }

    output[0] = xAcc / scoreAcc;
    output[1] = yAcc / scoreAcc;
    output[2] = sourcePtr[peakLocY*width + peakLocX];
}

void nmsCpu(float * targetPtr, int* kernelPtr, const float * const sourcePtr, const float threshold,
            const std::array<int, 4>& targetSize, const std::array<int, 4>& sourceSize)
{
    try
    {
        // Security checks
        if (sourceSize.empty())
            ;
        if (targetSize.empty())
            ;
        if (threshold < 0 || threshold > 1.0)
            ;

        // Params
        const auto channels = targetSize[1]; // 57
        const auto sourceHeight = sourceSize[2]; // 368
        const auto sourceWidth = sourceSize[3]; // 496
        const auto targetPeaks = targetSize[2]; // 97
        const auto targetPeakVec = targetSize[3]; // 3
        const auto sourceChannelOffset = sourceWidth * sourceHeight;
        const auto targetChannelOffset = targetPeaks * targetPeakVec;

        // Per channel operation
        for (auto c = 0 ; c < channels ; c++)
        {
            auto* currKernelPtr = &kernelPtr[c*sourceChannelOffset];
            const float * currSourcePtr = &sourcePtr[c*sourceChannelOffset];

            for (auto y = 0; y < sourceHeight; y++)
                for (auto x = 0; x < sourceWidth; x++)
                    nmsRegisterKernelCPU(currKernelPtr, currSourcePtr, sourceWidth, sourceHeight, threshold, x, y);

            auto currentPeakCount = 1;
            auto* currTargetPtr = &targetPtr[c*targetChannelOffset];
            for (auto y = 0; y < sourceHeight; y++)
            {
                for (auto x = 0; x < sourceWidth; x++)
                {
                    const auto index = y*sourceWidth + x;
                    // Find high intensity points
                    if (currentPeakCount < targetPeaks)
                    {
                        if (currKernelPtr[index] == 1)
                        {
                            // Accurate Peak Position
                            nmsAccuratePeakPosition(currSourcePtr, x, y, sourceWidth, sourceHeight,
                                                    &currTargetPtr[currentPeakCount*3]);
                            currentPeakCount++;
                        }
                    }

                }
            }
            currTargetPtr[0] = currentPeakCount-1;
        }
    }
    catch (const std::exception& e)
    {
        // error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
}

void PaintEleThickColor(cv::Mat img,cv::Point p1,cv::Point p2, int partindex , float thick,float maxdis)
{
    float distance = op::caldis(p1,p2);
    if(distance>(maxdis*4))
        return;
    if(partindex<=2 && distance>maxdis)
        return;

    cv::Point center=cv::Point((p1.x+p2.x)/2,(p1.y+p2.y)/2);
    cv::Scalar color;
    thick=thick/2;
    thick= fmin(fmax(thick,1),20);
    switch (partindex)
    {
        case 1:
            color=cv::Scalar(255,0,0);
            break;
        case 2:
            color=cv::Scalar(0,255,0);
            break;
        case 3:
            color=cv::Scalar(191,225,102);
            break;
        case 4:
            color=cv::Scalar(191,225,102);
            break;
        case 5:
            color=cv::Scalar(191,225,102);
            break;
        case 6:
            color=cv::Scalar(191,225,102);
            break;
        case 7:
            color=cv::Scalar(59,166,243);
            break;
        case 8:
            color=cv::Scalar(229,142,76);
            break;
        case 9:
            color=cv::Scalar(229,142,76);
            break;
    }
    cv::line(img,p1,p2,color,thick,16);
    cv::circle(img,p1,thick,cv::Scalar(241,46,154),thick*2);
    cv::circle(img,p2,thick,cv::Scalar(241,46,154),thick*2);
}


std::vector<float> NMSCON(const cv::Mat& img,float scale,int net_inh,int net_inw)
{
    int heightcount=(img.rows)/57;//46
    int widthcount=img.cols;//34

    float * cupdata=(float *)img.data;
    cv::Mat matheatmap=cv::Mat(net_inh * 57, net_inw,CV_32SC1);
    cv::Mat matPeaks = cv::Mat(97*57 , 3,CV_32SC1);
    int bigtotal=net_inw*net_inh;

    for(int i=0;i<57;i++)
    {
        cv::Mat mmb = cv::Mat(heightcount,widthcount, CV_32FC1, (float *)(cupdata+i*heightcount*widthcount ));
        cv::Mat mma = mmb.clone();


        cv::Mat mmd = cv::Mat(net_inh, net_inw,CV_32FC1);
        cv::resize(mma, mmd,cv::Size(net_inw,net_inh),0,0,CV_INTER_CUBIC);
        memcpy(((float *)(matheatmap.data))+i*bigtotal,(float *)(mmd.data),bigtotal*4);

        cv::Mat kernelmat = cv::Mat(net_inh, net_inw,CV_32SC1);
        cv::Mat targetmat = cv::Mat(97,3,CV_32SC1);

        std::array<int, 4> targetSize;
        targetSize[0]=1;
        targetSize[1]=1;
        targetSize[2]=97;
        targetSize[3]=3;
        std::array<int, 4> sourceSize;
        sourceSize[0]=1;
        sourceSize[1]=1;
        sourceSize[2]=net_inh;
        sourceSize[3]=net_inw;

        nmsCpu((float *)(targetmat.data), (int *)(kernelmat.data), (float *)(mmd.data), (float)0.03,targetSize, sourceSize);
        memcpy(((float *)(matPeaks.data))+i*97*3,(float *)(targetmat.data),97*3*4);

        //cv::imwrite("PeaksImage.png",matPeaks);
    }

    std::vector<float> poseKeypoints;
    std::vector<float> poseScores;

    //float scale=((float)(net_inw))/368.0;

    //std::cout<<"fdsafdsafdsafdsafdsfdsafdsafdsa"<< net_inw <<std::endl;


    cv::Point point;
    point.x=net_inw;
    point.y=net_inh;
    op::connectBodyPartsCpu(&poseKeypoints, &poseScores, (float * )matheatmap.data,(float *)matPeaks.data, point,96,0.95,0.05,3,0.4,scale);
    //    maxPeaks96
    //    mInterMinAboveThreshold0.95
    //    mInterThreshold0.05
    //    mMinSubsetCnt3
    //    mMinSubsetScore0.4
    //    mScaleNetToOutput1.76753

    //cv::imwrite("HeatMapImage.png",matheatmap);

    return poseKeypoints;
}

cv::Mat PaintImage(cv::Mat imageori,std::vector<float> poseKeypoints)
{
    float ms = 0.2;
    for(int ml = 0;ml<(poseKeypoints.size()/54);ml++)
    {
        int basecount=ml*3*18;
        float * basp= poseKeypoints.data() +basecount;

        cv::Point p1 = *(basp + 0*3+2)< ms? cv::Point(0,0):cv::Point(   (int)*(basp + 0*3   ),   (int)*(basp + 0*3 +1  ));
        cv::Point p2 = *(basp + 1*3+2)< ms? cv::Point(0,0):cv::Point(   (int)*(basp + 1*3   ),   (int)*(basp + 1*3 +1  ));
        cv::Point p3 = *(basp + 2*3+2)< ms? cv::Point(0,0):cv::Point(   (int)*(basp + 2*3   ),   (int)*(basp + 2*3 +1  ));
        cv::Point p4 = *(basp + 3*3+2)< ms? cv::Point(0,0):cv::Point(   (int)*(basp + 3*3   ),   (int)*(basp + 3*3 +1  ));
        cv::Point p5 = *(basp + 4*3+2)< ms? cv::Point(0,0):cv::Point(   (int)*(basp + 4*3   ),   (int)*(basp + 4*3 +1  ));
        cv::Point p6 = *(basp + 5*3+2)< ms? cv::Point(0,0):cv::Point(   (int)*(basp + 5*3   ),   (int)*(basp + 5*3 +1  ));
        cv::Point p7 = *(basp + 6*3+2)< ms? cv::Point(0,0):cv::Point(   (int)*(basp + 6*3   ),   (int)*(basp + 6*3 +1  ));
        cv::Point p8 = *(basp + 7*3+2)< ms? cv::Point(0,0):cv::Point(   (int)*(basp + 7*3   ),   (int)*(basp + 7*3 +1  ));
        cv::Point p9 = *(basp + 8*3+2)< ms? cv::Point(0,0):cv::Point(   (int)*(basp + 8*3   ),   (int)*(basp + 8*3 +1  ));
        cv::Point p10 = *(basp+ 9*3+2)< ms? cv::Point(0,0):cv::Point(   (int)*(basp + 9*3   ),   (int)*(basp + 9*3 +1  ));
        cv::Point p11 = *(basp+10*3+2)< ms? cv::Point(0,0):cv::Point(   (int)*(basp +10*3   ),   (int)*(basp +10*3 +1  ));
        cv::Point p12 = *(basp+11*3+2)< ms? cv::Point(0,0):cv::Point(   (int)*(basp +11*3   ),   (int)*(basp +11*3 +1  ));
        cv::Point p13 = *(basp+12*3+2)< ms? cv::Point(0,0):cv::Point(   (int)*(basp +12*3   ),   (int)*(basp +12*3 +1  ));
        cv::Point p14 = *(basp+13*3+2)< ms? cv::Point(0,0):cv::Point(   (int)*(basp +13*3   ),   (int)*(basp +13*3 +1  ));
        cv::Point p15 = *(basp+14*3+2)< ms? cv::Point(0,0):cv::Point(   (int)*(basp +14*3   ),   (int)*(basp +14*3 +1  ));
        cv::Point p16 = *(basp+15*3+2)< ms? cv::Point(0,0):cv::Point(   (int)*(basp +15*3   ),   (int)*(basp +15*3 +1  ));
        cv::Point p17 = *(basp+16*3+2)< ms? cv::Point(0,0):cv::Point(   (int)*(basp +16*3   ),   (int)*(basp +16*3 +1  ));
        cv::Point p18 = *(basp+17*3+2)< ms? cv::Point(0,0):cv::Point(   (int)*(basp +17*3   ),   (int)*(basp +17*3 +1  ));


        //大于平均长度4倍不要
        //脸部大于平均长度的不画
        //当个人连线少于等于6根不画

        int linecount=0;
        float dis=0;
        if(p1.x>0 && p15.x>0)
        {
            float distance = op::caldis(p1,p15);
            dis+=distance;
            linecount++;
        }
        if(p1.x>0 && p16.x>0)
        {
            float distance = op::caldis(p1,p16);
            dis+=distance;
            linecount++;
        }
        if(p1.x>0 && p2.x>0)
        {
            float distance = op::caldis(p1,p2);
            dis+=distance;
            linecount++;
        }
        if(p2.x>0 && p3.x>0)
        {
            float distance = op::caldis(p2,p3);
            dis+=distance;
            linecount++;
        }
        if(p2.x>0 && p6.x>0)
        {
            float distance = op::caldis(p2,p6);
            dis+=distance;
            linecount++;
        }
        if(p6.x>0 && p7.x>0)
        {
            float distance = op::caldis(p6,p7);
            dis+=distance;
            linecount++;
        }
        if(p3.x>0 && p4.x>0)
        {
            float distance = op::caldis(p3,p4);
            dis+=distance;
            linecount++;
        }
        if(p7.x>0 && p8.x>0)
        {
            float distance = op::caldis(p7,p8);
            dis+=distance;
            linecount++;
        }
        if(p4.x>0 && p5.x>0)
        {
            float distance = op::caldis(p4,p5);
            dis+=distance;
            linecount++;
        }
        if(p2.x>0 && p9.x>0)
        {
            float distance = op::caldis(p2,p9);
            dis+=distance;
            linecount++;
        }
        if(p12.x>0 && p2.x>0)
        {
            float distance = op::caldis(p12,p2);
            dis+=distance;
            linecount++;
        }
        if(p9.x>0 && p10.x>0)
        {
            float distance = op::caldis(p9,p10);
            dis+=distance;
            linecount++;
        }
        if(p12.x>0 && p13.x>0)
        {
            float distance = op::caldis(p12,p13);
            dis+=distance;
            linecount++;
        }
        if(p10.x>0 && p11.x>0)
        {
            float distance = op::caldis(p10,p11);
            dis+=distance;
            linecount++;
        }
        if(p13.x>0 && p14.x>0)
        {
            float distance = op::caldis(p13,p14);
            dis+=distance;
            linecount++;
        }
        if(p17.x>0 && p15.x>0)
        {
            float distance = op::caldis(p17,p15);
            dis+=distance;
            linecount++;
        }
        if(p16.x>0 && p18.x>0)
        {
            float distance = op::caldis(p16,p18);
            dis+=distance;
            linecount++;
        }
        dis=(dis/linecount);
        if(linecount<7)
        {
            //std::cout<<"line count < 7 continue"<<std::endl;
            continue;
        }



        float thick=dis/10;


//region paint all
//        if(p1.x>0 && p15.x>0)
//            PaintEleThick(imageori,p1,p15,  2,thick,dis);
//        if(p1.x>0 && p16.x>0)
//            PaintEleThick(imageori,p1,p16,  2,thick,dis);
//        if(p1.x>0 && p2.x>0)
//            PaintEleThick(imageori,p1,p2,  3,thick,dis);
//        if(p2.x>0 && p3.x>0)
//            PaintEleThick(imageori,p2,p3,  4,thick,dis);
//        if(p2.x>0 && p6.x>0)
//            PaintEleThick(imageori,p2,p6,  4,thick,dis);
//        if(p6.x>0 && p7.x>0)
//            PaintEleThick(imageori,p6,p7,  5,thick,dis);
//        if(p3.x>0 && p4.x>0)
//            PaintEleThick(imageori,p3,p4,  5,thick,dis);
//        if(p7.x>0 && p8.x>0)
//            PaintEleThick(imageori,p7,p8,  6,thick,dis);
//        if(p4.x>0 && p5.x>0)
//            PaintEleThick(imageori,p4,p5,  6,thick,dis);
//        if(p2.x>0 && p9.x>0)
//            PaintEleThick(imageori,p2,p9,  7,thick,dis);
//        if(p12.x>0 && p2.x>0)
//            PaintEleThick(imageori,p12,p2,  7,thick,dis);
//        if(p9.x>0 && p10.x>0)
//            PaintEleThick(imageori,p9,p10,  8,thick,dis);
//        if(p12.x>0 && p13.x>0)
//            PaintEleThick(imageori,p12,p13,  8,thick,dis);
//        if(p10.x>0 && p11.x>0)
//            PaintEleThick(imageori,p10,p11,  9,thick,dis);
//        if(p13.x>0 && p14.x>0)
//            PaintEleThick(imageori,p13,p14,  9,thick,dis);
//        if(p17.x>0 && p15.x>0)
//            PaintEleThick(imageori,p15,p17,  1,thick,dis);
//        if(p16.x>0 && p18.x>0)
//            PaintEleThick(imageori,p16,p18,  1,thick,dis);
//endregion paint all

        if(p3.x>0 && p9.x>0)//肩膀-髋骨
            PaintEleThickColor(imageori,p3,p9,  7,thick,dis);
        if(p6.x>0 && p12.x>0)//肩膀-髋骨
            PaintEleThickColor(imageori,p6,p12,  7,thick,dis);
        if(p12.x>0 && p9.x>0)//髋骨-髋骨
            PaintEleThickColor(imageori,p12,p9,  7,thick,dis);

        if(p9.x>0 && p10.x>0)//髋骨-膝盖
            PaintEleThickColor(imageori,p9,p10,  8,thick,dis);
        if(p12.x>0 && p13.x>0)//髋骨-膝盖
            PaintEleThickColor(imageori,p12,p13,  8,thick,dis);
        if(p10.x>0 && p11.x>0)//膝盖-脚腕
            PaintEleThickColor(imageori,p10,p11,  9,thick,dis);
        if(p13.x>0 && p14.x>0)//膝盖-脚腕
            PaintEleThickColor(imageori,p13,p14,  9,thick,dis);



        cv::Point head(0,0);
        if(p1.x>0)
        {
            head.x=p1.x;
            head.y=p1.y;
        }
        else if (p15.x>0 && p16.x>0)
        {
            head.x=(p15.x+p16.x)/2;
            head.y=(p15.y+p16.y)/2;
        }
        else if (p17.x>0 && p18.x>0)
        {
            head.x=(p17.x+p18.x)/2;
            head.y=(p17.y+p18.y)/2;
        }
        cv::Point neck(0,0);
        if(p2.x>0)
        {
            neck.x=p2.x;
            neck.y=p2.y;
        }
        else if (p3.x>0 && p6.x>0)
        {
            neck.x=(p3.x+p6.x)/2;
            neck.y=(p3.y+p6.y)/2;
        }

        if(neck.x>0 && head.x>0)//鼻子 脖子
            PaintEleThickColor(imageori,neck,head,  3,thick,dis);
        if(p2.x>0 && p3.x>0)//脖子-肩膀
            PaintEleThickColor(imageori,p2,p3,  4,thick,dis);
        if(p2.x>0 && p6.x>0)//脖子-肩膀
            PaintEleThickColor(imageori,p2,p6,  4,thick,dis);
        if(p6.x>0 && p7.x>0)// 5肩膀-手肘
            PaintEleThickColor(imageori,p6,p7,  5,thick,dis);
        if(p3.x>0 && p4.x>0)// 5肩膀-手肘
            PaintEleThickColor(imageori,p3,p4,  5,thick,dis);
        if(p7.x>0 && p8.x>0)// 5肩膀-手肘
            PaintEleThickColor(imageori,p7,p8,  6,thick,dis);
        if(p4.x>0 && p5.x>0)// 5肩膀-手肘
            PaintEleThickColor(imageori,p4,p5,  6,thick,dis);

        //1鼻子，2脖子，3右肩，4右肘，5右腕，6左肩，7左肘，8左腕，9右髋，10右膝，11右踝，12左髋，13左膝，14左踝，15左眼，16右眼，17左耳，18右耳，
        // 1眼睛-耳朵  2耳朵-鼻子  3鼻子-脖子  4脖子-肩膀  5肩膀-手肘  6手肘-手腕  7脖子-髋骨  8髋骨-膝盖  9膝盖-脚腕

    }


    //1鼻子，2脖子，3右肩，4右肘，5右腕，6左肩，7左肘，8左腕，9右髋，10右膝，11右踝，12左髋，13左膝，14左踝，15左眼，16右眼，17左耳，18右耳，

/*
    if(!(logo).empty() && imageori.rows==mask.rows && imageori.cols==mask.cols)
    {
        imageori=imageori-mask;
        imageori=logo+imageori;
    }
*/

    return imageori;
}



cv::Mat create_netsize_im(const Mat &im,const int netw,const int neth, float *scale)
{
    // for tall image
    int newh = neth;
    float s = newh / (float)im.rows;
    int neww = im.cols * s;
    if (neww > netw)
        {
        //for fat image
        neww = netw;
        s = neww / (float)im.cols;
        newh = im.rows * s;
        }

    *scale = 1 / s;
    Rect dst_area(0, 0, neww, newh);
    Mat dst = Mat::zeros(neth, netw, CV_8UC3);
    resize(im, dst(dst_area), Size(neww, newh));
    return dst;
}

