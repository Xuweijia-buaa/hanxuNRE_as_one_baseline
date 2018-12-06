#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <string>
#include <algorithm>

using namespace std;

//inPath: string

string inPath = "./data/";    //　装有train.txt,entity.txt.relation.txt
//string inPath = "../transX/";
//string inPath="../transX_embedding_hanxu_vec/";
//string inPath="../transX_embedding_word2vec_glove/";
//string inPath="../transX_embedding_hanxu_vec_keepwd/";
//string inPath="../transX_embedding_word2vec_glove_keepwd/";
//string inPath="../transX_embedding_glove840B300d_keepwd/";
extern "C"
void setInPath(char *path) {  // lib.setInPath("../data/")
    int len = strlen(path);
    inPath = "";
    for (int i = 0; i < len; i++)
        inPath = inPath + path[i];
    printf("Input Files Path : %s\n", inPath.c_str());
}

//　lefHead[e1] 每个e1在triple head中开始的位置　 0 4 5 9（第一个不用管，就是triple head[0].h,位置0）
// rigHead[e1] 每个e1在triple head中结束的位置   3 4 8 9 (include)(没有出现e1的位置,值为-1)
//　lefTail[e2] 每个e2在triple tail中开始的位置
//　rigTail[e2] 每个e2在triple tail中结束的位置　　(include)
//  没有出现e1/e2的位置. lefhead[3]＝０，righead[3]＝-1
int *lefHead, *rigHead;
int *lefTail, *rigTail;

struct Triple {
	int h, r, t;
};
// 包含()重载的类，成为函数对象类，作为Triple 数组的比较器。　默认小的排在前（导致return True的a在前） 重新定义ａ’小‘
struct cmp_head {
	bool operator()(const Triple &a, const Triple &b) {
		return (a.h < b.h)||(a.h == b.h && a.r < b.r)||(a.h == b.h && a.r == b.r && a.t < b.t);
	}
};

struct cmp_tail {
	bool operator()(const Triple &a, const Triple &b) {
		return (a.t < b.t)||(a.t == b.t && a.r < b.r)||(a.t == b.t && a.r == b.r && a.h < b.h);
	}
};

struct cmp_list {
	int minimal(int a,int b) {
		if (a > b) return b;
		return a;
	}
	bool operator()(const Triple &a, const Triple &b) {
		return (minimal(a.h, a.t) > minimal(b.h, b.t));
	}
};

Triple *trainHead, *trainTail, *trainList;
int relationTotal, entityTotal, tripleTotal;
int *freqRel, *freqEnt;
float *left_mean, *right_mean;

extern "C"
void init() {

	FILE *fin;
	int tmp;

	fin = fopen((inPath + "relation2id.txt").c_str(), "r");
	tmp = fscanf(fin, "%d", &relationTotal);
	fclose(fin);
	printf("%d\n", relationTotal);

	freqRel = (int *)calloc(relationTotal, sizeof(int));
	
	fin = fopen((inPath + "entity2id.txt").c_str(), "r");
	tmp = fscanf(fin, "%d", &entityTotal);
	fclose(fin);
	printf("%d\n", entityTotal);

	freqEnt = (int *)calloc(entityTotal, sizeof(int));
	
	fin = fopen((inPath + "triple2id.txt").c_str(), "r");
	tmp = fscanf(fin, "%d", &tripleTotal);
	printf("%d\n", tripleTotal);
	trainHead = (Triple *)calloc(tripleTotal, sizeof(Triple));    // trainHead:（0,0,0),（0,0,1),（0,0,2) 
	trainTail = (Triple *)calloc(tripleTotal, sizeof(Triple));    // trainTail:（0,0,0),（1,0,0),（2,0,0)
	trainList = (Triple *)calloc(tripleTotal, sizeof(Triple));    // trainList[i]: all original triple in train_file
	tripleTotal = 0;
	while (fscanf(fin, "%d", &trainList[tripleTotal].h) == 1) {
		tmp = fscanf(fin, "%d", &trainList[tripleTotal].t);
		tmp = fscanf(fin, "%d", &trainList[tripleTotal].r);
		freqEnt[trainList[tripleTotal].t]++;                      // 根据triple idx, 记录entity,relation的出现次数。
		freqEnt[trainList[tripleTotal].h]++;
		freqRel[trainList[tripleTotal].r]++;
		trainHead[tripleTotal].h = trainList[tripleTotal].h;
		trainHead[tripleTotal].t = trainList[tripleTotal].t;
		trainHead[tripleTotal].r = trainList[tripleTotal].r;
		trainTail[tripleTotal].h = trainList[tripleTotal].h;
		trainTail[tripleTotal].t = trainList[tripleTotal].t;
		trainTail[tripleTotal].r = trainList[tripleTotal].r;
		tripleTotal++;
	}
	fclose(fin);

	sort(trainHead, trainHead + tripleTotal, cmp_head());       // 按照e1,e2的idx,分别重新排列了所有triple.
	sort(trainTail, trainTail + tripleTotal, cmp_tail());       // void sort (RandomAccessIterator first, RandomAccessIterator last, Compare comp);

	lefHead = (int *)calloc(entityTotal, sizeof(int));
	rigHead = (int *)calloc(entityTotal, sizeof(int));
	lefTail = (int *)calloc(entityTotal, sizeof(int));
	rigTail = (int *)calloc(entityTotal, sizeof(int));
	memset(rigHead, -1, sizeof(rigHead));
	memset(rigTail, -1, sizeof(rigTail));
	for (int i = 1; i < tripleTotal; i++) {
		if (trainTail[i].t != trainTail[i - 1].t) {
			rigTail[trainTail[i - 1].t] = i - 1;
			lefTail[trainTail[i].t] = i;
		}
		if (trainHead[i].h != trainHead[i - 1].h) {
			rigHead[trainHead[i - 1].h] = i - 1;
			lefHead[trainHead[i].h] = i;
		}
	}
	rigHead[trainHead[tripleTotal - 1].h] = tripleTotal - 1;
	rigTail[trainTail[tripleTotal - 1].t] = tripleTotal - 1;

    //每个关系在所有出现总次数/不同e1下该关系出现的次数(多个属于同一e1的triple,r只看做出现了１次)
    // 如果不同ｒ出现在不同e1中，该值为１;如果所有r都出现在同一e1中，该值很大。
	// 表明该关系是否与某e1联系紧密.该关系分布越集中，该值越大.否则该值接近１．分布较分散，与e1无关
	left_mean = (float *)calloc(relationTotal,sizeof(float));
	right_mean = (float *)calloc(relationTotal,sizeof(float));
	for (int i = 0; i < entityTotal; i++) {
		for (int j = lefHead[i] + 1; j < rigHead[i]; j++)
			if (trainHead[j].r != trainHead[j - 1].r)
				left_mean[trainHead[j].r] += 1.0;
		if (lefHead[i] <= rigHead[i])
			left_mean[trainHead[lefHead[i]].r] += 1.0;
		for (int j = lefTail[i] + 1; j < rigTail[i]; j++)
			if (trainTail[j].r != trainTail[j - 1].r)
				right_mean[trainTail[j].r] += 1.0;
		if (lefTail[i] <= rigTail[i])
			right_mean[trainTail[lefTail[i]].r] += 1.0;
	}
	for (int i = 0; i < relationTotal; i++) {
		left_mean[i] = freqRel[i] / left_mean[i];    //每个关系在所有出现总次数/不同e1下该关系出现的次数(多个属于同一e1的triple,r只看做出现了１次).该值很大,r与某e1关系密切
		right_mean[i] = freqRel[i] / right_mean[i];
	}
}

extern "C"
int getEntityTotal() {
	return entityTotal;
}

extern "C"
int getRelationTotal() {
	return relationTotal;
}

extern "C"
int getTripleTotal() {
	return tripleTotal;
}

// unsigned long long *next_random;
unsigned long long next_random = 3;

unsigned long long randd(int id) {
	next_random = next_random * (unsigned long long)25214903917 + 11;
	return next_random;
}
// 返回属于thread id的伪随机数，范围为0-ｘ
int rand_max(int id, int x) {
	int res = randd(id) % x;
	while (res<0)
		res+=x;
	return res;
}
//返回不与e1,r构成triple的e2　idx
int corrupt_head(int id, int h, int r) {
	int lef, rig, mid, ll, rr;
	lef = lefHead[h] - 1;
	rig = rigHead[h];
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainHead[mid].r >= r) rig = mid; else
		lef = mid;
	}
	ll = rig;
	lef = lefHead[h];
	rig = rigHead[h] + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainHead[mid].r <= r) lef = mid; else
		rig = mid;
	}
	rr = lef;
	int tmp = rand_max(id, entityTotal - (rr - ll + 1));
	if (tmp < trainHead[ll].t) return tmp;
	if (tmp > trainHead[rr].t - rr + ll - 1) return tmp + rr - ll + 1;
	lef = ll, rig = rr + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainHead[mid].t - mid + ll - 1 < tmp)
			lef = mid;
		else 
			rig = mid;
	}
	return tmp + lef - ll + 1;
}

int corrupt_tail(int id, int t, int r) {
	int lef, rig, mid, ll, rr;
	lef = lefTail[t] - 1;
	rig = rigTail[t];
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainTail[mid].r >= r) rig = mid; else
		lef = mid;
	}
	ll = rig;
	lef = lefTail[t];
	rig = rigTail[t] + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainTail[mid].r <= r) lef = mid; else
		rig = mid;
	}
	rr = lef;
	int tmp = rand_max(id, entityTotal - (rr - ll + 1));
	if (tmp < trainTail[ll].h) return tmp;
	if (tmp > trainTail[rr].h - rr + ll - 1) return tmp + rr - ll + 1;
	lef = ll, rig = rr + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainTail[mid].h - mid + ll - 1 < tmp)
			lef = mid;
		else 
			rig = mid;
	}
	return tmp + lef - ll + 1;
}
// 返回kb的这个batch 的所有样本i的本身idx, neg_sample idx
extern "C"
void getBatch(int *ph, int *pt, int *pr, int *nh, int *nt, int *nr, int batchSize, int id = 0) {　// id，没有用到。只有一个线程
	// ph: np数组的首地址，指向连续int区域
	for (int batch = 0; batch < batchSize; batch++) {
		int i = rand_max(id, tripleTotal), j;　　　// i: 0-T之间的伪随机数　不同batch应该每次用不同i.除非每次调用，next_random状态保持，i才不同
		// 或者直接500
		//该值越大， 该triple的ｒ,和某些特定e2越密切
		float prob = 1000 * right_mean[trainList[i].r] / (right_mean[trainList[i].r] + left_mean[trainList[i].r]);
		// batch里的每一个样本。如果每个batch next_random都从头来，那么只有batch不同样本不一样。所有batch 就都一样了
		if (randd(id) % 1000 < prob) {  //　不替换r,只替换e1/e2 
		    // 每个样本i，相同e1,r,不同e2
			j = corrupt_head(id, trainList[i].h, trainList[i].r);
			ph[batch] = trainList[i].h;
			pt[batch] = trainList[i].t;
			pr[batch] = trainList[i].r;
			nh[batch] = trainList[i].h;
			nt[batch] = j;
			nr[batch] = trainList[i].r;
		} else {
			// 每个样本i，相同e2,r,不同e1
			j = corrupt_tail(id, trainList[i].t, trainList[i].r);
			ph[batch] = trainList[i].h;
			pt[batch] = trainList[i].t;
			pr[batch] = trainList[i].r;
			nh[batch] = j;
			nt[batch] = trainList[i].t;
			nr[batch] = trainList[i].r;
		}
	}
}