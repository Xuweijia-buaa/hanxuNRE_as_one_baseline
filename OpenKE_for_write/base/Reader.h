#ifndef READER_H
#define READER_H
#include "Setting.h"
#include "Triple.h"
#include <cstdlib>
#include <algorithm>
#include "PreTriple.h"//1
#include <iostream>  
#include <string>  
#include <vector>  
#include <fstream>  

INT *freqRel, *freqEnt;
INT *lefHead, *rigHead;
INT *lefTail, *rigTail;
INT *lefRel, *rigRel;
REAL *left_mean, *right_mean;

Triple *trainList;
Triple *trainHead;
Triple *trainTail;
Triple *trainRel;

INT *testLef, *testRig;
INT *validLef, *validRig;

extern "C"
void importTrainFiles() {

	printf("The toolkit is importing datasets.\n");
	FILE *fin;
	int tmp;

	fin = fopen((inPath + "relation2id.txt").c_str(), "r");
	tmp = fscanf(fin, "%ld", &relationTotal);　//值赋给relationTotal
	printf("The total of relations is %ld.\n", relationTotal);
	fclose(fin);

    // 打开entity文件，只读取了数目
	fin = fopen((inPath + "entity2id.txt").c_str(), "r");
	tmp = fscanf(fin, "%ld", &entityTotal);
	printf("The total of entities is %ld.\n", entityTotal);//值赋给 entityTotal
	fclose(fin);

        // 打开train2id文件
	fin = fopen((inPath + "train2id.txt").c_str(), "r");
	tmp = fscanf(fin, "%ld", &trainTotal); //值赋给trainTotal

	trainList = (Triple *)calloc(trainTotal, sizeof(Triple));
	trainHead = (Triple *)calloc(trainTotal, sizeof(Triple));　// 按e1排序  e1,r,e2
	trainTail = (Triple *)calloc(trainTotal, sizeof(Triple));　// 按e2排序　e2 r,e1
	trainRel = (Triple *)calloc(trainTotal, sizeof(Triple));　　//  e1,e2,r
　
	freqRel = (INT *)calloc(relationTotal, sizeof(INT));
	freqEnt = (INT *)calloc(entityTotal, sizeof(INT));//contral probablity of break head /tail

	for (INT i = 0; i < trainTotal; i++) {
		tmp = fscanf(fin, "%ld", &trainList[i].h);
		tmp = fscanf(fin, "%ld", &trainList[i].t);
		tmp = fscanf(fin, "%ld", &trainList[i].r);
	}
	fclose(fin);
        // 按照triple中e1,r,e2的id　从小到大顺序排序　in_place　e1,r,e2 小的在前
	std::sort(trainList, trainList + trainTotal, Triple::cmp_head);

        // 
	tmp = trainTotal; trainTotal = 1;
	trainHead[0] = trainTail[0] = trainRel[0] = trainList[0];　//第一个都放首个triple
	freqEnt[trainList[0].t] += 1;
	freqEnt[trainList[0].h] += 1;
	freqRel[trainList[0].r] += 1;
	for (INT i = 1; i < tmp; i++)
　　　　　　　　　　　　　　　　// 从第2个triple开始，只要有一个元素不同于前一个triple，
		if (trainList[i].h != trainList[i - 1].h || trainList[i].r != trainList[i - 1].r || trainList[i].t != trainList[i - 1].t) {
                        // 去重，同时把这个triple赋给trainHead,trainTail,trainRel
			trainHead[trainTotal] = trainTail[trainTotal] = trainRel[trainTotal] = trainList[trainTotal] = trainList[i];
			trainTotal++;
                        // 统计entity,relation出现次数 　　　　　　trainList[i].h : e1的id　　　　　　freqEnt[trainList[i].h]  e1位置
			freqEnt[trainList[i].t]++;
			freqEnt[trainList[i].h]++;
			freqRel[trainList[i].r]++;
		}

	std::sort(trainHead, trainHead + trainTotal, Triple::cmp_head); //e1,r,e2 小的在前
	std::sort(trainTail, trainTail + trainTotal, Triple::cmp_tail); //e2,r,e1
	std::sort(trainRel, trainRel + trainTotal, Triple::cmp_rel);    //e1,e2,r
	printf("The total of train triples is %ld.\n", trainTotal);

        // 总entity个
	lefHead = (INT *)calloc(entityTotal, sizeof(INT));//对于每个entity, 按照head从小到大，lefHead，rigHead[e_id]对应的head是这个entity的最左和最右的triple id (在trainHea中)
	rigHead = (INT *)calloc(entityTotal, sizeof(INT));
	lefTail = (INT *)calloc(entityTotal, sizeof(INT));//对于每个entity, 按照tail从小到大，lefTail，rigTail[e_id]对应的tail是这个entity的最左和最右的triple id（在trainTail中）
	rigTail = (INT *)calloc(entityTotal, sizeof(INT));


	lefRel = (INT *)calloc(entityTotal, sizeof(INT));//对于每个entity, 按照e1,e2,r　　tail从小到大，lefTail，rigTail[e_id]对应的tail是这个entity的最左和最右的triple id（在trainTail中）
	rigRel = (INT *)calloc(entityTotal, sizeof(INT));

　　　　　　　　//初始化为-1　void *memset(void *s, int ch, size_t n); 将s中当前位置后面的n个字节 （typedef unsigned int size_t ）用 ch 替换并返回 s 
	memset(rigHead, -1, sizeof(INT)*entityTotal);
	memset(rigTail, -1, sizeof(INT)*entityTotal);
	memset(rigRel, -1, sizeof(INT)*entityTotal);

	for (INT i = 1; i < trainTotal; i++) {
		if (trainTail[i].t != trainTail[i - 1].t) { //trainTail(e2,r,e1): [*,*,3],[*,*,4],[*,*,4],[*,*,6],[*,*,9]
                                                            //                   i: 0        1       2       3      4
			rigTail[trainTail[i - 1].t] = i - 1;  // i=1,   rigTail[3]=0   lefTail[4]=1  　　　i=2,pass     i=3  rigTail[4]=2  lefTail[6]=3     i=4  rigTail[6]=3,lefTail[9]=4 
			lefTail[trainTail[i].t] = i;
		}
		if (trainHead[i].h != trainHead[i - 1].h) {
			rigHead[trainHead[i - 1].h] = i - 1;
			lefHead[trainHead[i].h] = i;
		}
		if (trainRel[i].h != trainRel[i - 1].h) {　　// 
			rigRel[trainRel[i - 1].h] = i - 1;
			lefRel[trainRel[i].h] = i;
		}
	}
	lefHead[trainHead[0].h] = 0;
	rigHead[trainHead[trainTotal - 1].h] = trainTotal - 1;
	lefTail[trainTail[0].t] = 0;
	rigTail[trainTail[trainTotal - 1].t] = trainTotal - 1;
	lefRel[trainRel[0].h] = 0;
	rigRel[trainRel[trainTotal - 1].h] = trainTotal - 1;

	left_mean = (REAL *)calloc(relationTotal,sizeof(REAL));
	right_mean = (REAL *)calloc(relationTotal,sizeof(REAL));
	for (INT i = 0; i < entityTotal; i++) {
		for (INT j = lefHead[i] + 1; j < rigHead[i]; j++)
			if (trainHead[j].r != trainHead[j - 1].r)
				left_mean[trainHead[j].r] += 1.0;
		if (lefHead[i] <= rigHead[i])
			left_mean[trainHead[lefHead[i]].r] += 1.0;
		for (INT j = lefTail[i] + 1; j < rigTail[i]; j++)
			if (trainTail[j].r != trainTail[j - 1].r)
				right_mean[trainTail[j].r] += 1.0;
		if (lefTail[i] <= rigTail[i])
			right_mean[trainTail[lefTail[i]].r] += 1.0;
	}
	for (INT i = 0; i < relationTotal; i++) {
		left_mean[i] = freqRel[i] / left_mean[i];
		right_mean[i] = freqRel[i] / right_mean[i];
	}
}

// SAMPLE
Triple *testList;
Triple *validList;
Triple *tripleList;

extern "C"
void importTestFiles() {
    FILE *fin;
    INT tmp;
    
	fin = fopen((inPath + "relation2id.txt").c_str(), "r");
    tmp = fscanf(fin, "%ld", &relationTotal);
    fclose(fin);

	fin = fopen((inPath + "entity2id.txt").c_str(), "r");
    tmp = fscanf(fin, "%ld", &entityTotal);
    fclose(fin);

    FILE* f_kb1 = fopen((inPath + "test2id.txt").c_str(), "r");//1 to 1
    FILE* f_kb2 = fopen((inPath + "train2id.txt").c_str(), "r");
    FILE* f_kb3 = fopen((inPath + "valid2id.txt").c_str(), "r");
    tmp = fscanf(f_kb1, "%ld", &testTotal);
    tmp = fscanf(f_kb2, "%ld", &trainTotal);
    tmp = fscanf(f_kb3, "%ld", &validTotal);
    tripleTotal = testTotal + trainTotal + validTotal;//总的ｔｒｉｐｌｅ数目
    testList = (Triple *)calloc(testTotal, sizeof(Triple));
    validList = (Triple *)calloc(validTotal, sizeof(Triple));
    tripleList = (Triple *)calloc(tripleTotal, sizeof(Triple));
    for (INT i = 0; i < testTotal; i++) {
        tmp = fscanf(f_kb1, "%ld", &testList[i].h);
        tmp = fscanf(f_kb1, "%ld", &testList[i].t);
        tmp = fscanf(f_kb1, "%ld", &testList[i].r);
        tripleList[i] = testList[i];
    }
    for (INT i = 0; i < trainTotal; i++) {
        tmp = fscanf(f_kb2, "%ld", &tripleList[i + testTotal].h);
        tmp = fscanf(f_kb2, "%ld", &tripleList[i + testTotal].t);
        tmp = fscanf(f_kb2, "%ld", &tripleList[i + testTotal].r);
    }
    for (INT i = 0; i < validTotal; i++) {
        tmp = fscanf(f_kb3, "%ld", &tripleList[i + testTotal + trainTotal].h);
        tmp = fscanf(f_kb3, "%ld", &tripleList[i + testTotal + trainTotal].t);
        tmp = fscanf(f_kb3, "%ld", &tripleList[i + testTotal + trainTotal].r);
        validList[i] = tripleList[i + testTotal + trainTotal];
    }
    fclose(f_kb1);
    fclose(f_kb2);
    fclose(f_kb3);

    std::sort(tripleList, tripleList + tripleTotal, Triple::cmp_head);
    std::sort(testList, testList + testTotal, Triple::cmp_rel2);
    std::sort(validList, validList + validTotal, Triple::cmp_rel2);
    printf("The total of test triples is %ld.\n", testTotal);
    printf("The total of valid triples is %ld.\n", validTotal);

    testLef = (INT *)calloc(relationTotal, sizeof(INT));
	testRig = (INT *)calloc(relationTotal, sizeof(INT));
	memset(testLef, -1, sizeof(INT)*relationTotal);
	memset(testRig, -1, sizeof(INT)*relationTotal);
	for (INT i = 1; i < testTotal; i++) {
		if (testList[i].r != testList[i-1].r) {
			testRig[testList[i-1].r] = i - 1;
			testLef[testList[i].r] = i;
		}
	}
	testLef[testList[0].r] = 0;
	testRig[testList[testTotal - 1].r] = testTotal - 1;


	validLef = (INT *)calloc(relationTotal, sizeof(INT));
	validRig = (INT *)calloc(relationTotal, sizeof(INT));
	memset(validLef, -1, sizeof(INT)*relationTotal);
	memset(validRig, -1, sizeof(INT)*relationTotal);
	for (INT i = 1; i < validTotal; i++) {
		if (validList[i].r != validList[i-1].r) {
			validRig[validList[i-1].r] = i - 1;
			validLef[validList[i].r] = i;
		}
	}
	validLef[validList[0].r] = 0;
	validRig[validList[validTotal - 1].r] = validTotal - 1;
}

// read  line getline
 // NEW GLOBAL to predict
 // new structure
 // Triple int ,int ,set 
// Predict
std::vector<std::string> split(std::string& str, const char* c)//2
{
	char *cstr, *p;
	std::vector<std::string> res;
	cstr = new char[str.size() + 1];
	strcpy(cstr, str.c_str());
	p = strtok(cstr, c);
	while (p != NULL)
	{
		res.push_back(p);
		p = strtok(NULL, c);
	}
	return res;
}

Pre_Triple *predict_testList;  //3
Pre_Triple *predict_validList;
extern "C"
void import_predict_TestFiles() {
    FILE *fin;
    INT tmp;
	// no problem
	fin = fopen((inPath + "relation2id.txt").c_str(), "r");
    tmp = fscanf(fin, "%ld", &relationTotal);
    fclose(fin);

	fin = fopen((inPath + "entity2id.txt").c_str(), "r");
    tmp = fscanf(fin, "%ld", &entityTotal);
    fclose(fin);

	std::ifstream file(inPath + "original_test2id.txt", std::ios::in);// 4 test
	INT n = 0;
	INT testTotal_pre;
	std::string textline;
	//string delim("/t");
	std::vector<std::string> results; // store split result
	if (file.good())
	{
		while (!file.eof() && file.peek() != EOF)
		{
			getline(file, textline);      //while(fin.getline(line, sizeof(line)))
			std::cout << "all: " << textline << std::endl;
			if (n == 0) {
				testTotal_pre = atoi(textline.c_str());
				predict_testList = (Pre_Triple *)calloc(testTotal_pre, sizeof(Pre_Triple));
			}
			else {
				// split   n:1-total    index:0-total-1
				results = split(textline, "\t");
				for (INT i = 0; i < results.size(); i++) {
					if (i == 0) {
						predict_testList[n - 1].h = atoi(results[i].c_str());
						std::cout << "h : " << predict_testList[n - 1].h << std::endl;
					}
					else if (i == 1) {
						predict_testList[n - 1].r = atoi(results[i].c_str());
						std::cout << "r : " << predict_testList[n - 1].r << std::endl;
					}
					else {
						predict_testList[n - 1].vec.push_back(atoi(results[i].c_str()));
					}
				}
				for (INT i = 0; i < predict_testList[n - 1].vec.size(); i++) {
					std::cout << "n : " << n - 1 << "t : " << predict_testList[n - 1].vec[i] << std::endl;
				}
			}
			n += 1;
		}
	}
	file.close();

	std::ifstream file(inPath + "original_valid2id.txt", std::ios::in);// 4 valid
	INT n = 0;
	INT validTotal_pre;
	std::string textline;
	//string delim("/t");
	std::vector<std::string> results; // store split result
	if (file.good())
	{
		while (!file.eof() && file.peek() != EOF)
		{
			getline(file, textline);      //while(fin.getline(line, sizeof(line)))
			std::cout << "all: " << textline << std::endl;
			if (n == 0) {
				validTotal_pre = atoi(textline.c_str());
				predict_validList = (Pre_Triple *)calloc(validTotal_pre, sizeof(Pre_Triple));
			}
			else {
				// split   n:1-total    index:0-total-1
				results = split(textline, "\t");
				for (INT i = 0; i < results.size(); i++) {
					if (i == 0) {
						predict_validList[n - 1].h = atoi(results[i].c_str());
						std::cout << "h : " << predict_validList[n - 1].h << std::endl;
					}
					else if (i == 1) {
						predict_validList[n - 1].r = atoi(results[i].c_str());
						std::cout << "r : " << predict_validList[n - 1].r << std::endl;
					}
					else {
						predict_validList[n - 1].vec.push_back(atoi(results[i].c_str()));
					}
				}
				for (INT i = 0; i <predict_validList[n - 1].vec.size(); i++) {
					std::cout << "n : " << n - 1 << "t : " << predict_validList[n - 1].vec[i] << std::endl;
				}
			}
			n += 1;
		}
	}
	file.close();
    printf("The total of predict test triples is %ld.\n", testTotal_pre);
    printf("The total of predict valid triples is %ld.\n", validTotal_pre);
}

INT head_lef[10000];
INT head_rig[10000];
INT tail_lef[10000];
INT tail_rig[10000];
INT head_type[1000000];
INT tail_type[1000000];

extern "C"
void importTypeFiles() {
	INT total_lef = 0;
    INT total_rig = 0;
    FILE* f_type = fopen((inPath + "type_constrain.txt").c_str(),"r");
    INT tmp;
    tmp = fscanf(f_type, "%ld", &tmp);
    for (INT i = 0; i < relationTotal; i++) {
        INT rel, tot;
        tmp = fscanf(f_type, "%ld%ld", &rel, &tot);
        head_lef[rel] = total_lef;
        for (INT j = 0; j < tot; j++) {
            tmp = fscanf(f_type, "%ld", &head_type[total_lef]);
            total_lef++;
        }
        head_rig[rel] = total_lef;
        std::sort(head_type + head_lef[rel], head_type + head_rig[rel]);
        tmp = fscanf(f_type, "%ld%ld", &rel, &tot);
        tail_lef[rel] = total_rig;
        for (INT j = 0; j < tot; j++) {
            tmp = fscanf(f_type, "%ld", &tail_type[total_rig]);
            total_rig++;
        }
        tail_rig[rel] = total_rig;
        std::sort(tail_type + tail_lef[rel], tail_type + tail_rig[rel]);
    }
    fclose(f_type);
}


#endif
