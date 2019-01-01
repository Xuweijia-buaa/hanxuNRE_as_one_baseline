#define REAL float
#define INT int

#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <string>
#include <algorithm>
#include <pthread.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

using namespace std;

const REAL pi = 3.141592653589793238462643383;

INT threads = 8;
INT bernFlag = 0;
INT loadBinaryFlag = 0;
INT outBinaryFlag = 0;
INT trainTimes = 1000;  // epoch 50000/100000
INT nbatches = 1;       // how many batches each epoch    Batch: batch size=n_triple/nbatches
INT dimension = 100;
REAL alpha = 0.001;
REAL margin = 1.0;

string inPath = "./";
string outPath = "";
string loadPath = "";
string note = "";

INT *lefHead, *rigHead;
INT *lefTail, *rigTail;
// Triple define, a struct
struct Triple {
	INT h, r, t;
};

Triple *trainHead, *trainTail, *trainList; // Triple *　类型的地址。　Triple数组数组名/数组首地址　　存放的元素是Triple类型

// 包含()重载的类，成为函数对象类，作为Triple 数组的比较器。　默认小的排在前（导致return True的a在前） 重新定义ａ’小‘
struct cmp_head {　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　// e1在前重排 (r,e2)
	bool operator()(const Triple &a, const Triple &b) {　　　　　　　　　　　// a,b:对数组内元素的常引用
		return (a.h < b.h)||(a.h == b.h && a.r < b.r)||(a.h == b.h && a.r == b.r && a.t < b.t);
	}
};

struct cmp_tail {
	bool operator()(const Triple &a, const Triple &b) {　　　　　　　　　　// e2在前重排 　(r,e1)
		return (a.t < b.t)||(a.t == b.t && a.r < b.r)||(a.t == b.t && a.r == b.r && a.h < b.h);
	}
};

/*
	There are some math functions for the program initialization.
*/
unsigned long long *next_random;

unsigned long long randd(INT id) {
	next_random[id] = next_random[id] * (unsigned long long)25214903917 + 11;   //thread id 对应的下一个伪随机数。伪随机数生成公式
	return next_random[id];
}

INT rand_max(INT id, INT x) {  // x:T
	INT res = randd(id) % x;   // 返回属于thread id的伪随机数范围为0-T
	while (res<0)
		res+=x;
	return res;
}
// return [min-max)
REAL rand(REAL min, REAL max) {
	return min + (max - min) * rand() / (RAND_MAX + 1.0);  // int rand (void): can return any random value betewwn 0-RAND_MAX
}                                                          // rand() / (RAND_MAX + 1.0): (0,1)

REAL normal(REAL x, REAL miu,REAL sigma) {
	return 1.0/sqrt(2*pi)/sigma*exp(-1*(x-miu)*(x-miu)/(2*sigma*sigma)); // return x's probability density  under normal distribution m,sigma
}

// 返回min-max之间，以较高概率处在该正太分布中心处的值　　　　大于　self-define randn function
REAL randn(REAL miu,REAL sigma, REAL min ,REAL max) {
	REAL x, y, dScope;
	do {
		x = rand(min,max);        // x: between [min-max)   self-reload rand 
		y = normal(x,miu,sigma);  // x　位置处的概率密度  　　　　self define normal.
		dScope=rand(0.0,normal(miu,miu,sigma));  // dScope:a value between [0-该分布的最高概率密度)
	} while (dScope > y);
	return x;
}

//传入的是地址（指针），vec的值被直接改变
void norm(REAL * con) {
	REAL x = 0;
	for (INT  ii = 0; ii < dimension; ii++)
		x += (*(con + ii)) * (*(con + ii));　// *(con + ii): vector con 的第ii个元素
	x = sqrt(x);
	// 只有该vector |x|2 大于１，才会normalize
	if (x>1)
		for (INT ii=0; ii < dimension; ii++)
			*(con + ii) /= x;　　　　　　　　　　//该vector 每个元素，值被归一化. 
}

/*
	Read triples from the training file.
*/

INT relationTotal, entityTotal, tripleTotal;
REAL *relationVec, *entityVec;　　　　　　　　　　　　// float 数组　　数组名/数组首地址
REAL *relationVecDao, *entityVecDao;
INT *freqRel, *freqEnt;
REAL *left_mean, *right_mean;

void init() {

	FILE *fin;
	INT tmp;
　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　// c_str: 把string 对象转换成c中的字符串样式,return char *
                                                               //FILE * fopen ( const char * filename, const char * mode );
	fin = fopen((inPath + "relation2id.txt").c_str(), "r");　　// inPath  由命令行参数char *, 赋值给string. string 可以加字符串常量
	tmp = fscanf(fin, "%d", &relationTotal);　　　　　　        // fscanf遇到空格和换行时结束. get relationTotal. 将输入读到变量所在地址　　　　
	fclose(fin);                                              // c_str(): 把string 对象转换成c中的字符串样式

	relationVec = (REAL *)calloc(relationTotal * dimension, sizeof(REAL));   //分配|R|*h 个长度为sizeof(float)的连续空间,并随机初始化
	for (INT i = 0; i < relationTotal; i++) {
		for (INT ii=0; ii<dimension; ii++)
			relationVec[i * dimension + ii] = randn(0, 1.0 / dimension, -6 / sqrt(dimension), 6 / sqrt(dimension)); // U |6/sqrt(h)|   接近　N(0,1/h)
　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　//  relation vec初始并不做归一化
	fin = fopen((inPath + "entity2id.txt").c_str(), "r");                    // get |E|
	tmp = fscanf(fin, "%d", &entityTotal);
	fclose(fin);

	entityVec = (REAL *)calloc(entityTotal * dimension, sizeof(REAL));       //分配|E|*h 个长度为sizeof(float)的连续空间,并随机初始化
	for (INT i = 0; i < entityTotal; i++) {
		for (INT ii=0; ii<dimension; ii++)
			entityVec[i * dimension + ii] = randn(0, 1.0 / dimension, -6 / sqrt(dimension), 6 / sqrt(dimension));
		norm(entityVec+i*dimension);                                         // entity_vec[i] 所在地址。　
	}　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　// 如果模值大于１，entity　vec 对应的向量归一化

	freqRel = (INT *)calloc(relationTotal + entityTotal, sizeof(INT)); // 前一半是ｒ fre, 后一半是　e　fre  (|E+R|个连续空间,int   freqRel:数组名＝＝数组地址。)
	freqEnt = freqRel + relationTotal;                                 //　freqEnt：　真实entity的地址

	fin = fopen((inPath + "train2id.txt").c_str(), "r");
	tmp = fscanf(fin, "%d", &tripleTotal);　　　　　　　　　　　　　　　　// get n_triple
	//能存放T个triple的Triple数组。
	trainHead = (Triple *)calloc(tripleTotal, sizeof(Triple));　　　　// trainHead:（0,0,0),（0,0,1),（0,0,2) 
	trainTail = (Triple *)calloc(tripleTotal, sizeof(Triple));　　　　// trainTail:（0,0,0),（1,0,0),（2,0,0)
	trainList = (Triple *)calloc(tripleTotal, sizeof(Triple));      // trainList[i]: all original triple in train_file
	for (INT i = 0; i < tripleTotal; i++) {
		tmp = fscanf(fin, "%d", &trainList[i].h);
		tmp = fscanf(fin, "%d", &trainList[i].t);
		tmp = fscanf(fin, "%d", &trainList[i].r);
		freqEnt[trainList[i].t]++;　　　　　　　　　　　　　　　　　　　　　// 根据triple idx, 记录entity,relation的出现次数。
		freqEnt[trainList[i].h]++;                                   // freqEnt[e_idx]: e的出现次数
		freqRel[trainList[i].r]++;　　　　　　　　　　　　　　　　　　　　　// freqRel[r_idx]: r的出现次数　freqEnt占用freqRel[|R|:]之后的内存
		trainHead[i] = trainList[i];　　　　　　　　　　　　　　　
		trainTail[i] = trainList[i];
	}
	fclose(fin);

	sort(trainHead, trainHead + tripleTotal, cmp_head());　　　　　　　// void sort (RandomAccessIterator first, RandomAccessIterator last, Compare comp);
	sort(trainTail, trainTail + tripleTotal, cmp_tail());           // 按照e1,e2的idx,分别重新排列了所有triple.

																   // |E|  初始均为０                
																   // 记录每一个e1的triple在triple_head中的span位置                 
	lefHead = (INT *)calloc(entityTotal, sizeof(INT));             //　lefHead[e1] 每个e1在triple head中开始的位置　 0 4 5 9（第一个不用管，就是triple head[0].h,位置0）
	rigHead = (INT *)calloc(entityTotal, sizeof(INT));　　　　　　　　// rigHead[e1] 每个e1在triple head中结束的位置   3 4 8 9 (include)(没有出现e1的位置,值为-1)
	lefTail = (INT *)calloc(entityTotal, sizeof(INT));             //　lefTail[e2] 每个e2在triple tail中开始的位置
	rigTail = (INT *)calloc(entityTotal, sizeof(INT));             //　rigTail[e2] 每个e2在triple tail中结束的位置　　(include)(没有出现e2的位置,值为-1)
	memset(rigHead, -1, sizeof(INT)*entityTotal);　　　　　　　　　　　// rig span，没有出现e1的地方 set to -1  比如３不在e1中. lefhead[3]＝０，righead[3]＝-1
	memset(rigTail, -1, sizeof(INT)*entityTotal);
	for (INT i = 1; i < tripleTotal; i++) {
		if (trainTail[i].t != trainTail[i - 1].t) {　　　//相邻的triple，e2不相等，
			rigTail[trainTail[i - 1].t] = i - 1;　　　　　//前一个triple的e2　　在triple tail中结束的位置
			lefTail[trainTail[i].t] = i;　　　　　　　　　　//后一个triple的e2　　在triple tail中开始的位置
		}
		if (trainHead[i].h != trainHead[i - 1].h) {　　　//相邻的triple，e1不相等
			rigHead[trainHead[i - 1].h] = i - 1;        //该e1　在triple head中结束的位置　（该位置triple仍然是e1,下一个triple不是） 0-T-1(最早结束在0)
			lefHead[trainHead[i].h] = i;　　　　　　　　　 //该e1　在triple head中开始的位置, （该位置triple是e1,之前不是）　　　　　　　1-T-1 (最早的改变出现在1)
		}　　　　　　　　　　　　　　　　　　　　　　　　　　　　// 没有出现e1 entity的位置，值为-1.  lefthead[0]=0
	}
	rigHead[trainHead[tripleTotal - 1].h] = tripleTotal - 1; // 最后一个triple对应的e1, 在triple head中结束的位置:T-1
	rigTail[trainTail[tripleTotal - 1].t] = tripleTotal - 1;

	left_mean = (REAL *)calloc(relationTotal * 2, sizeof(REAL)); // 2|R|,前一半是left_mean，后一半是right_mean.
	right_mean = left_mean + relationTotal;                      // 每个e1对应的不同r的数目
	for (INT i = 0; i < entityTotal; i++) {
		for (INT j = lefHead[i] + 1; j <= rigHead[i]; j++)  // e1的每一对相邻triple，包含不同r的数目,后一r统计+1. 把该e1下的所有r看做只出现了一次，不重复统计多个triple
			if (trainHead[j].r != trainHead[j - 1].r)　　　　  
				left_mean[trainHead[j].r] += 1.0;　　　　　　
		if (lefHead[i] <= rigHead[i])　　　　　　　　　　　　　// 不存在的，lefHead[e]=0,rigHead[1]=-1,不满足。满足该条件的ｅ均为当过e1的e,其范围总是left-rig [0-4],[5-5]
			left_mean[trainHead[lefHead[i]].r] += 1.0;    // 该e1的初始r，统计+1
		for (INT j = lefTail[i] + 1; j <= rigTail[i]; j++)
			if (trainTail[j].r != trainTail[j - 1].r)
				right_mean[trainTail[j].r] += 1.0;
		if (lefTail[i] <= rigTail[i])
			right_mean[trainTail[lefTail[i]].r] += 1.0;
	}
　　　// |R|.表明某关系是否与某e1联系紧密。该值越大，分布越集中
	for (INT i = 0; i < relationTotal; i++) {
		left_mean[i] = freqRel[i] / left_mean[i];　　　　　　//每个关系在所有出现总次数/不同e1下该关系出现的次数(多个属于同一e1的triple,r只看做出现了１次)
		right_mean[i] = freqRel[i] / right_mean[i];　　　　 // 如果不同ｒ出现在不同e1中，该值为１;如果所有r都出现在同一e1中，该值很大。
	}　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　// 表明该关系是否与某e1联系紧密.该关系分布越集中，该值越大.否则该值接近１．分布较分散，与e1无关

	relationVecDao = (REAL*)calloc(dimension * relationTotal, sizeof(REAL));　// 保存vec的导数
	entityVecDao = (REAL*)calloc(dimension * entityTotal, sizeof(REAL));
}
// load binary pretained vec
void load_binary() {
	struct stat statbuf1;
	if (stat((loadPath + "entity2vec" + note + ".bin").c_str(), &statbuf1) != -1) {  // &statbuf1:指针，stat函数获取file_name指向文件的文件状态，并将文件信息保存到结构体buf中
		//statbuf1.st_size:文件大小, 以字节计算
		INT fd = open((loadPath + "entity2vec" + note + ".bin").c_str(), O_RDONLY);
		//建立内存映射,)用来将某个文件内容映射到内存中，对该内存区域的存取即是直接对该文件内容的读写。
		// 映射区的开始地址/映射区的长度(以字节为单位，不足一内存页按一内存页处理)/页内容可以被读取
		//MAP_PRIVATE 建立一个写入时拷贝的私有映射。内存区域的写入不会影响到原文件
		// 有效的文件描述词,(open返回)/ 被映射对象内容的起点
		// 成功时，返回被映射区的指针
		REAL* entityVecTmp = (REAL*)mmap(NULL, statbuf1.st_size, PROT_READ, MAP_PRIVATE, fd, 0); 
		memcpy(entityVec, entityVecTmp, statbuf1.st_size);
		munmap(entityVecTmp, statbuf1.st_size);  //取消entityVecTmp所指的内存映射
		close(fd);
	}  
	struct stat statbuf2;
	if (stat((loadPath + "relation2vec" + note + ".bin").c_str(), &statbuf2) != -1) {  
		INT fd = open((loadPath + "relation2vec" + note + ".bin").c_str(), O_RDONLY);
		REAL* relationVecTmp =(REAL*)mmap(NULL, statbuf2.st_size, PROT_READ, MAP_PRIVATE, fd, 0); 
		memcpy(relationVec, relationVecTmp, statbuf2.st_size);
		munmap(relationVecTmp, statbuf2.st_size);
		close(fd);
	}
}
// load pretained vec
void load() {
	if (loadBinaryFlag) {
		load_binary();
		return;
	}
	FILE *fin;
	INT tmp;
	fin = fopen((loadPath + "entity2vec" + note + ".vec").c_str(), "r");　// loadpath:string. +后c_str()，char *
	for (INT i = 0; i < entityTotal; i++) {
		INT last = i * dimension;                    // i's vector's start pos in entityVec数组　0,h,2h,...ih
		for (INT j = 0; j < dimension; j++)
			tmp = fscanf(fin, "%f", &entityVec[last + j]); // each element j in this vector, give it to entityVec[i*h + j]
	}
	fclose(fin);
	fin = fopen((loadPath + "relation2vec" + note + ".vec").c_str(), "r");
	for (INT i = 0; i < relationTotal; i++) {
		INT last = i * dimension;
		for (INT j = 0; j < dimension; j++)
			tmp = fscanf(fin, "%f", &relationVec[last + j]);
	}
	fclose(fin);
}


/*
	Training process of transE.
*/

INT Len;
INT Batch;
REAL res;

REAL calc_sum(INT e1, INT e2, INT rel) {
	REAL sum=0;
	INT last1 = e1 * dimension;      // e1在entityVec的起始位置（地址）
	INT last2 = e2 * dimension;
	INT lastr = rel * dimension;
	for (INT ii=0; ii < dimension; ii++)
	　  // |e2-(e1+r)|
		sum += fabs(entityVec[last2 + ii] - entityVec[last1 + ii] - relationVec[lastr + ii]);
	return sum;
}

void gradient(INT e1_a, INT e2_a, INT rel_a, INT e1_b, INT e2_b, INT rel_b) {
	INT lasta1 = e1_a * dimension;
	INT lasta2 = e2_a * dimension;
	INT lastar = rel_a * dimension;
	INT lastb1 = e1_b * dimension;
	INT lastb2 = e2_b * dimension;
	INT lastbr = rel_b * dimension;
	for (INT ii=0; ii  < dimension; ii++) {
		REAL x;
		x = (entityVec[lasta2 + ii] - entityVec[lasta1 + ii] - relationVec[lastar + ii]); // grad for negsample
		// each element's loss, want it to be 0, * a
		if (x > 0)
			x = -alpha;
		else
			x = alpha;

		relationVec[lastar + ii] -= x;
		entityVec[lasta1 + ii] -= x;  
		entityVec[lasta2 + ii] += x;  // +负梯度。ｘ>0,-/ x<0,+

		x = (entityVec[lastb2 + ii] - entityVec[lastb1 + ii] - relationVec[lastbr + ii]);// grad for negsample
		if (x > 0)
			x = alpha;
		else
			x = -alpha;
		relationVec[lastbr + ii] -=  x;
		entityVec[lastb1 + ii] -= x;
		entityVec[lastb2 + ii] += x;  // +正梯度，希望ｘ增大
	}
}

void train_kb(INT e1_a, INT e2_a, INT rel_a, INT e1_b, INT e2_b, INT rel_b) {
	REAL sum1 = calc_sum(e1_a, e2_a, rel_a);  // transe score for this ex (smaller,better)
	REAL sum2 = calc_sum(e1_b, e2_b, rel_b);  // transe score for this ex's neg sample
	if (sum1 + margin > sum2) {               // score2-score1<margin,bp
		res += margin + sum1 - sum2;          // loss
		gradient(e1_a, e2_a, rel_a, e1_b, e2_b, rel_b);  // input, triple's idx。　这个triple和对应的neg sample ,bp
	}
}

//返回不与e1,r构成triple的e2　idx
INT corrupt_head(INT id, INT h, INT r) { // id: 该thread id     h,r:该进程该时刻选择的triple,对应的e1,r.
	INT lef, rig, mid, ll, rr;
	lef = lefHead[h] - 1;                // 所有e1开始的triple在trainHead对应的位置　（lef,rig]
	rig = rigHead[h];
	while (lef + 1 < rig) {              //
		mid = (lef + rig) >> 1;          // trainHead[mid]：　e1对应的triple的中间位置triple的关系r idx大于随机选择的这个triple的r idx (r在前半部分，留下前半部分)
		if (trainHead[mid].r >= r) rig = mid; else　　// (r在前半部分，留下前半部分,否则留下后半部分。如果与该triple的r相同，一直压缩到该span，与r相同triple的最左边pos　
		lef = mid;
	}
	ll = rig;                           // 直到到达该triple在trainHead中相同r所在的位置的left
	lef = lefHead[h];
	rig = rigHead[h] + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainHead[mid].r <= r) lef = mid; else
		rig = mid;
	}
	rr = lef;　　　　　　　　　　　　　　　　// 直到到达该triple在trainHead中相同r所在的位置的left    trainHead[ll]-trainHead[rr], 和该triple有相同e1,r。
	　　　　　　　　　　　　　　　　　　　　　// ((rr - ll + 1):e1，r对应的triple最多包涵的e2数目（e2_idx:6,8,18）最多包含3个e2
	INT tmp = rand_max(id, entityTotal - (rr - ll + 1)); // 产生0-E 之间随机数。因为该随机数有+n_triples操作，所以产生 0-E-n_triples之间的随机数  
	if (tmp < trainHead[ll].t) return tmp;　　　　　　　 　// 该e2不和这个triple有相同e1,r,直接返回　tmp<6　　　trainHead[ll].t：　e1,r对应的所有triple中，t_idx最小的triple. 
	if (tmp > trainHead[rr].t - rr + ll - 1) return tmp + rr - ll + 1;// 该e2+n_triples后，范围也不在e1,r包含的t里 tmp＋(3)>18. 直接返回tmp＋(3)
	// 如果tmp 对应的e2 idx,，既大于６，又小于15
	// [6,8,10,13,15,18]   tmp:9, out:12(9+3)/ tmp:10, out:14(10+4) / tmp:11, out:16(11+5)
	lef = ll, rig = rr + 1;           // 
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainHead[mid].t - mid + ll - 1 < tmp) // tmp+  (mid + ll - 1) mid之前的元素数目（包括mid处）
			lef = mid;
		else 
			rig = mid;
	}
	return tmp + lef - ll + 1;  // still in span, 但比某个mid.t的值大。
}

INT corrupt_tail(INT id, INT t, INT r) {
	INT lef, rig, mid, ll, rr;
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
	INT tmp = rand_max(id, entityTotal - (rr - ll + 1));
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

void* trainMode(void *con) {  // con: (long,thread id, 被强制转化为void *)，
	INT id, pr, i, j;
	id = (unsigned long long)(con);  //再被强制转化为unsigned long long
	next_random[id] = rand();        //  线程id的随机数：　0 and RAND_MAX　随机数数组 next_random,global
	// 当前epoch的当前某一线程id
	for (INT k = Batch / threads; k >= 0; k--) {　　// k:T/8-0    (Batch: T/1=T,global) all triple split into 8 threads
	    // 根据线程号，该线程随机从Ｔ中抽取样本，每次抽取一个，抽取T/8次　
		// 每个样本的neg sample 只corrupt一边，满足相同e1,r或者相同r,e2。另一边从非e1,r里随机取。取完正负两个样本立刻bp）
		i = rand_max(id, Len);　　　　　　　　　　　　　// i:thread id对应的随机数，用来选择一个样本　　[0-T)   Len=T，global
		if (bernFlag)　　　　
		　　 //该值越大， 该triple的ｒ,和某些特定e2越密切　　　　　　　　　　　　
			pr = 1000 * right_mean[trainList[i].r] / (right_mean[trainList[i].r] + left_mean[trainList[i].r]); // bern:尽量避免corrupt到正例
		else
			pr = 500;
		//　不替换r,只替换e1/e2 这个triple和对应的neg sample ,bp
		if (randd(id) % 1000 < pr) {　　//randd(id) ：任意一个0-1000的伪随机数。　pr越大，r和e2联系紧密，（如r只和e2一起出现） 替换e2，越不容易碰上正例
			j = corrupt_head(id, trainList[i].h, trainList[i].r);　　　　　　　// 返回某个不属于e1.r的e2. 或者接近最后的e2.
			train_kb(trainList[i].h, trainList[i].t, trainList[i].r, trainList[i].h, j, trainList[i].r);
		} else {
			j = corrupt_tail(id, trainList[i].t, trainList[i].r);
			train_kb(trainList[i].h, trainList[i].t, trainList[i].r, j, trainList[i].t, trainList[i].r);
		}
		// 该向量模sqrt(平方和)大于１，才norm
		norm(relationVec + dimension * trainList[i].r);  // r 在数组relationVec中的起始地址　0+ｈ*r_idx
		norm(entityVec + dimension * trainList[i].h);    // e1
		norm(entityVec + dimension * trainList[i].t);    // e2
		norm(entityVec + dimension * j);                 // neg e1/e2
	}
	pthread_exit(NULL);//结束该线程
}
/*int main()
    pthread_t tids[NUM_THREADS];// 定义线程的标识符数组.　数组内每个元素是pthread_t类型，用来标识thread id
    for(int i = 0; i < NUM_THREADS; ++i)
    {
        //参数依次是：创建的线程id(指向线程标识符的指针)　/ 线程参数 /调用的函数 / 传入的函数参数(把引用作为指针强制转换为 void 类型进行传递)  传入(void *)&(indexes[i])/NULL
        pthread_create(&tids[i], NULL, say_hello, NULL);　　　　　　　　　// 传入以后，再用int tid = *((int*)threadid)，由无类型指针变为整形数指针，然后再读取
    }
    //等各个线程退出后，进程才结束，否则进程强制结束了，线程可能还没反应过来；
    pthread_exit(NULL);//俩次运行的不同之处在于有没有这一行          // 通过 pthread_exit() 退出，mian进程本身结束后,其他线程将继续执行　/
	                                                           //  void pthread_exit(void* retval); 使子线程退出，并返回一个空指针类型的值。
	} */            


void* train(void *con) {
	Len = tripleTotal;
	Batch = Len / nbatches;　// Batch: T  nbatches:1
	next_random = (unsigned long long *)calloc(threads, sizeof(unsigned long long));  // global,指向unsigned long long类型的指针　含有n_threads个元素,均为０ 
	for (INT epoch = 0; epoch < trainTimes; epoch++) { //each epoch
		res = 0;// global para, float
		for (INT batch = 0; batch < nbatches; batch++) {  // just one time
			pthread_t * pt = (pthread_t *)malloc(threads * sizeof(pthread_t));　// pthread_t类型的数组，用来标识thread id
			for (long a = 0; a < threads; a++)
				pthread_create(&pt[a], NULL, trainMode,  (void*)a);　// 创建子线程。　　trainMode:子线程函数名　　　(void*)a：该线程id
			for (long a = 0; a < threads; a++)
				pthread_join(pt[a], NULL);　　//int pthread_join(pthread_t thread, void **retval); thread: 被连接线程的线程号
				                             // 当调用 pthread_join() 时，当前线程会处于阻塞状态(主线程)，直到被调用的线程pt[ｉ]结束后，主线程才会重新开始执行
											 // 没有pthread_join主线程会很快结束从而使整个进程结束，从而使创建的线程没有机会开始执行就结束了。
											 // 加入pthread_join后，主线程会一直等待直到等待的线程结束自己才结束，使创建的线程有机会执行
											 // 主线程在pthread_join(0,NULL);这里就挂起了，在等待0号线程结束后再等待1号线程。
											 // 如果345先结束，主线程等012结束后，发现345其实早已经结束了，就会回收345的资源
			// 该batch结束＝＝该epoch结束(该epoch的多个线程都结束了），回收malloc分配的内存（值为随机数）　pt=NULL;指针用完赋值NULL是一个很好的习惯。
			free(pt);
		}
		printf("epoch %d %f\n", epoch, res);
	}
}

/*
	Get the results of transE.
*/

void out_binary() {
		INT len, tot;
		REAL *head;		
		FILE* f2 = fopen((outPath + "relation2vec" + note + ".bin").c_str(), "wb");// wb
		FILE* f3 = fopen((outPath + "entity2vec" + note + ".bin").c_str(), "wb");
		len = relationTotal * dimension; tot = 0; // len:|R|h
		head = relationVec;
		while (tot < len) {
			//size_t fwrite ( const void * ptr, size_t size, size_t count, FILE * stream );
			// ptr:读入的起始地址
			//　size:要写入的每个元素字节数，Size in bytes
			// count:读入的元素数目
			// return:The total number of elements successfully written 
			INT sum = fwrite(head + tot, sizeof(REAL), len - tot, f2);  //  len - tot,还未写入的元素数目。希望都写入，如果不行，记录写入了多少，之后剩下的继续写
			tot = tot + sum;                                            //已经写入的元素数目（flaot）
		}
		len = entityTotal * dimension; tot = 0;
		head = entityVec;
		while (tot < len) {
			INT sum = fwrite(head + tot, sizeof(REAL), len - tot, f3);
			tot = tot + sum;
		}	
		fclose(f2);
		fclose(f3);
}

void out() {
		if (outBinaryFlag) {
			out_binary(); 
			return;
		}
		FILE* f2 = fopen((outPath + "relation2vec" + note + ".vec").c_str(), "w");
		FILE* f3 = fopen((outPath + "entity2vec" + note + ".vec").c_str(), "w");
		for (INT i=0; i < relationTotal; i++) {
			INT last = dimension * i;　　　　　　　　　　　　　　　　// 第i个vector首地址　h*i
			for (INT ii = 0; ii < dimension; ii++)             // h*i+j, 用"\t"分隔
				fprintf(f2, "%.6f\t", relationVec[last + ii]);　
			fprintf(f2,"\n");　　　　　　　　　　　　　　　　　　　　　// vector间分隔"\n"
		}
		for (INT  i = 0; i < entityTotal; i++) {
			INT last = i * dimension;
			for (INT ii = 0; ii < dimension; ii++)
				fprintf(f3, "%.6f\t", entityVec[last + ii] );
			fprintf(f3,"\n");
		}
		fclose(f2);
		fclose(f3);
}

/*
	Main function
*/

//字符串常量类型　：char *
int ArgPos(char *str, int argc, char **argv) {
	int a;
	for (a = 1; a < argc; a++) 
	if (!strcmp(str, argv[a])) {   // 如果匹配　　　　（equal，返回0.　/ first bigger, return >0）
		if (a == argc - 1) {
			printf("Argument missing for %s\n", str);
			exit(1);　　　　　　　　　// 如果匹配了但已经是最后一个参数了，　说明某个参数给了命令但没有赋值，进程退出
		}
		return a;                  // argv[i]: 第i个命令行参数, 和str (char *)"-output"之类匹配, 返回该命令行参数位置
	}
	return -1;                     // or return -1
}


void setparameters(int argc, char **argv) {
	int i;
	if ((i = ArgPos((char *)"-size", argc, argv)) > 0) dimension = atoi(argv[i + 1]);　　// ｉ位置，命令行参数和‘size’匹配，size value是argv[i+1]
	if ((i = ArgPos((char *)"-input", argc, argv)) > 0) inPath = argv[i + 1];           // 可以用　argv[i]  (char *) 对 path(string) 赋值
	if ((i = ArgPos((char *)"-output", argc, argv)) > 0) outPath = argv[i + 1];
	if ((i = ArgPos((char *)"-load", argc, argv)) > 0) loadPath = argv[i + 1];
	if ((i = ArgPos((char *)"-thread", argc, argv)) > 0) threads = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-epochs", argc, argv)) > 0) trainTimes = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-nbatches", argc, argv)) > 0) nbatches = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-margin", argc, argv)) > 0) margin = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-load-binary", argc, argv)) > 0) loadBinaryFlag = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-out-binary", argc, argv)) > 0) outBinaryFlag = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-note", argc, argv)) > 0) note = argv[i + 1];
}
//argc:输入的命令行参数个数　int  可执行程序程序本身的文件名，也算一个命令行参数，因此，argc 的值至少是1.   transe -size 100 -thread 8  : argc=5
//argv 是一个数组(指针数组)，其中的每个元素都是一个char* 类型的指针，argv[i]指针指向一个字符串，这个字符串里就存放着命令行参数  
//argv[0]:char *,指向该transe.exe 本身的文件名transe
int main(int argc, char **argv) { //char * (*argv) == char*  (argv[])
	setparameters(argc, argv);
	init();
	if (loadPath != "") load(); //默认为“”。　pretrained
	train(NULL);                // 用了多线程，每个epoch 用8个线程。跑完后，下一个epoch回收资源，分配新的８个线程
	if (outPath != "") out();
	return 0;
}
