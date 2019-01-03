#ifndef TEST_H
#define TEST_H
#include "Setting.h"
#include "Reader.h"
#include "Corrupt.h"
#include<numeric> 
/*=====================================================================================
link prediction
======================================================================================*/
INT lastHead = 0;
INT lastTail = 0;
REAL l1_filter_tot = 0, l1_tot = 0, r1_tot = 0, r1_filter_tot = 0, l_tot = 0, r_tot = 0, l_filter_rank = 0, l_rank = 0;
REAL l3_filter_tot = 0, l3_tot = 0, r3_tot = 0, r3_filter_tot = 0, l_filter_tot = 0, r_filter_tot = 0, r_filter_rank = 0, r_rank = 0;

extern "C"
void getHeadBatch(INT *ph, INT *pt, INT *pr) {
    for (INT i = 0; i < entityTotal; i++) {
        ph[i] = i;
        pt[i] = testList[lastHead].t;
        pr[i] = testList[lastHead].r;
    }
}

extern "C"
//1 fix h,r   predict t
// new structure testList  complete 1 e1,r
// only h,r
void getTailBatch(INT *ph, INT *pt, INT *pr) {
    for (INT i = 0; i < entityTotal; i++) {
		ph[i] = predict_testList[lastTail].h; // predict e1 id
		pr[i] = predict_testList[lastTail].r; // predict r id
		pt[i] = i;                            // all e2 	
       // ph[i] = testList[lastTail].h;
       // pt[i] = i;
       // pr[i] = testList[lastTail].r;
    }
}

extern "C"
void testHead(REAL *con) {
    INT h = testList[lastHead].h;
    INT t = testList[lastHead].t;
    INT r = testList[lastHead].r;

    REAL minimal = con[h];
    INT l_s = 0;
    INT l_filter_s = 0;
    INT l_s_constrain = 0;

    for (INT j = 0; j <= entityTotal; j++) {
        REAL value = con[j];
        if (j != h && value < minimal) {
            l_s += 1;
            if (not _find(j, t, r))
                l_filter_s += 1;
        }
    }

    if (l_filter_s < 10) l_filter_tot += 1;
    if (l_s < 10) l_tot += 1;
    if (l_filter_s < 3) l3_filter_tot += 1;
    if (l_s < 3) l3_tot += 1;
    if (l_filter_s < 1) l1_filter_tot += 1;
    if (l_s < 1) l1_tot += 1;
    l_filter_rank += (l_filter_s+1);
    l_rank += (1+l_s);
    lastHead++;
    printf("l_filter_s: %ld\n", l_filter_s);
    printf("%f %f %f %f\n", l_tot / lastHead, l_filter_tot / lastHead, l_rank / lastHead, l_filter_rank / lastHead);
}


// 2 con 1*E_total float 
// con 对应每个id处得到的每一个分数 应该从小到大排输出前1,2 index 看是否在vec中 float 数组
extern "C"
void testTail(REAL *con,INT *exist) {
    INT h = predict_testList[lastTail].h; //  e1 id
	INT r = predict_testList[lastTail].r; //  r  id
	std::vector<INT> ans_list;
    ans_list=predict_testList[lastTail].vec; // ans list entity id 
	int top1 = 0;
	int top2 = !top1;
	exist=0;
   // sort index
    INT len = entityTotal;
	std::vector<REAL> vec_con(con,con + len);// turn  score con into vector
	std::vector<INT> idx(vec_con.size());    // score index
	std::iota(idx.begin(), idx.end(), 0);    // init  index
	std::sort(idx.begin(), idx.end(),        // sort  index
		[&vec_con](INT i1, INT i2) {return vec_con[i1] < vec_con[i2]; });//compare funcion
	
	std::vector<INT>::iterator it;// find index
	
	if (top1) {
		INT predict=idx[0];
		std::cout << "top 1" << std::endl;
		it = find(ans_list.begin(), ans_list.end(), predict);
		if (it != ans_list.end()) {
			exist=1;
		}
	}
	if(top2){
		std::cout << "top 2" << std::endl;
		if (idx[0]==h){
			INT predict=idx[0];
		}else{
			INT predict=idx[1];
		}
		it = find(ans_list.begin(), ans_list.end(), predict);
		if (it != ans_list.end()) {
			exist=1;
		}
	}
    lastTail++;
    printf("r_filter_s: %ld\n", r_filter_s);
    printf("%f %f %f %f\n", r_tot /lastTail, r_filter_tot /lastTail, r_rank /lastTail, r_filter_rank /lastTail);
}

extern "C"
void test_link_prediction() {
    printf("overall results:\n");
    printf("left %f %f %f %f \n", l_rank/ testTotal, l_tot / testTotal, l3_tot / testTotal, l1_tot / testTotal);
    printf("left(filter) %f %f %f %f \n", l_filter_rank/ testTotal, l_filter_tot / testTotal,  l3_filter_tot / testTotal,  l1_filter_tot / testTotal);
    printf("right %f %f %f %f \n", r_rank/ testTotal, r_tot / testTotal,r3_tot / testTotal,r1_tot / testTotal);
    printf("right(filter) %f %f %f %f\n", r_filter_rank/ testTotal, r_filter_tot / testTotal,r3_filter_tot / testTotal,r1_filter_tot / testTotal);
}

/*=====================================================================================
triple classification
======================================================================================*/
Triple *negTestList;
extern "C"
void getNegTest() {
    negTestList = (Triple *)calloc(testTotal, sizeof(Triple));
    for (INT i = 0; i < testTotal; i++) {
        negTestList[i] = testList[i];
        negTestList[i].t = corrupt(testList[i].h, testList[i].r);
    }
    FILE* fout = fopen((inPath + "test_neg.txt").c_str(), "w");
    for (INT i = 0; i < testTotal; i++) {
        fprintf(fout, "%ld\t%ld\t%ld\t%ld\n", testList[i].h, testList[i].t, testList[i].r, INT(1));
        fprintf(fout, "%ld\t%ld\t%ld\t%ld\n", negTestList[i].h, negTestList[i].t, negTestList[i].r, INT(-1));
    }
    fclose(fout);
}

Triple *negValidList;
extern "C"
void getNegValid() {
    negValidList = (Triple *)calloc(validTotal, sizeof(Triple));
    for (INT i = 0; i < validTotal; i++) {
        negValidList[i] = validList[i];
        negValidList[i].t = corrupt(validList[i].h, validList[i].r);
    }
    FILE* fout = fopen((inPath + "valid_neg.txt").c_str(), "w");
    for (INT i = 0; i < validTotal; i++) {
        fprintf(fout, "%ld\t%ld\t%ld\t%ld\n", validList[i].h, validList[i].t, validList[i].r, INT(1));
        fprintf(fout, "%ld\t%ld\t%ld\t%ld\n", negValidList[i].h, negValidList[i].t, negValidList[i].r, INT(-1));
    }
    fclose(fout);
        
}

extern "C"
void getTestBatch(INT *ph, INT *pt, INT *pr, INT *nh, INT *nt, INT *nr) {
    getNegTest();
    for (INT i = 0; i < testTotal; i++) {
        ph[i] = testList[i].h;
        pt[i] = testList[i].t;
        pr[i] = testList[i].r;
        nh[i] = negTestList[i].h;
        nt[i] = negTestList[i].t;
        nr[i] = negTestList[i].r;
    }
}

extern "C"
void getValidBatch(INT *ph, INT *pt, INT *pr, INT *nh, INT *nt, INT *nr) {
    getNegValid();
    for (INT i = 0; i < validTotal; i++) {
        ph[i] = validList[i].h;
        pt[i] = validList[i].t;
        pr[i] = validList[i].r;
        nh[i] = negValidList[i].h;
        nt[i] = negValidList[i].t;
        nr[i] = negValidList[i].r;
    }
}

REAL *relThresh;
REAL threshEntire;
extern "C"
void getBestThreshold(REAL *score_pos, REAL *score_neg) {
    REAL interval = 0.01;
    relThresh = (REAL *)calloc(relationTotal, sizeof(REAL));
    REAL min_score, max_score, bestThresh, tmpThresh, bestAcc, tmpAcc;
    INT n_interval, correct, total;
    for (INT r = 0; r < relationTotal; r++) {
        if (validLef[r] == -1) continue;
        total = (validRig[r] - validLef[r] + 1) * 2;
        min_score = score_pos[validLef[r]];
        if (score_neg[validLef[r]] < min_score) min_score = score_neg[validLef[r]];
        max_score = score_pos[validLef[r]];
        if (score_neg[validLef[r]] > max_score) max_score = score_neg[validLef[r]];
        for (INT i = validLef[r]+1; i <= validRig[r]; i++) {
            if(score_pos[i] < min_score) min_score = score_pos[i];
            if(score_pos[i] > max_score) max_score = score_pos[i];
            if(score_neg[i] < min_score) min_score = score_neg[i];
            if(score_neg[i] > max_score) max_score = score_neg[i];
        }
        n_interval = INT((max_score - min_score)/interval);
        for (INT i = 0; i <= n_interval; i++) {
            tmpThresh = min_score + i * interval;
            correct = 0;
            for (INT j = validLef[r]; j <= validRig[r]; j++) {
                if (score_pos[j] <= tmpThresh) correct ++;
                if (score_neg[j] > tmpThresh) correct ++;
            }
            tmpAcc = 1.0 * correct / total;
            if (i == 0) {
                bestThresh = tmpThresh;
                bestAcc = tmpAcc;
            } else if (tmpAcc > bestAcc) {
                bestAcc = tmpAcc;
                bestThresh = tmpThresh;
            }
        }
        relThresh[r] = bestThresh;
        printf("relation %ld: bestThresh is %lf, bestAcc is %lf\n", r, bestThresh, bestAcc);
    }
}

REAL *testAcc;
REAL aveAcc;
extern "C"
void test_triple_classification(REAL *score_pos, REAL *score_neg) {
    testAcc = (REAL *)calloc(relationTotal, sizeof(REAL));
    INT aveCorrect = 0, aveTotal = 0;
    REAL aveAcc;
    for (INT r = 0; r < relationTotal; r++) {
        if (validLef[r] == -1 || testLef[r] ==-1) continue;
        INT correct = 0, total = 0;
        for (INT i = testLef[r]; i <= testRig[r]; i++) {
            if (score_pos[i] <= relThresh[r]) correct++;
            if (score_neg[i] > relThresh[r]) correct++;
            total += 2;
        }
        testAcc[r] = 1.0 * correct / total;
        aveCorrect += correct; 
        aveTotal += total;
        printf("relation %ld: triple classification accuracy is %lf\n", r, testAcc[r]);
    }
    aveAcc = 1.0 * aveCorrect / aveTotal;
    printf("average accuracy is %lf\n", aveAcc);
}

#endif
