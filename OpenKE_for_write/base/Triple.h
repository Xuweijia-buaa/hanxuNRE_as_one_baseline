#ifndef TRIPLE_H
#define TRIPLE_H
#include "Setting.h"

struct Triple {

	INT h, r, t;

        // 返回int a，int b小值 int
	static INT minimal(INT a,INT b) {
		if (a > b) return b;
		return a;
	}
	// Triple a,b 
        // triple a中entity 较小值　是否大于　triple b中entity 较小值
        // a大 返回bool　True
	static bool cmp_list(const Triple &a, const Triple &b) {
		return (minimal(a.h, a.t) > minimal(b.h, b.t));
	}
        // 优先比较triple a,b 的e1 ——>e1小 True  比较顺序　e1,r,e2
	static bool cmp_head(const Triple &a, const Triple &b) {
		return (a.h < b.h)||(a.h == b.h && a.r < b.r)||(a.h == b.h && a.r == b.r && a.t < b.t);
	}
　　　　　　　　// 优先比较triple a,b 的e2 ——>e2小 True 比较顺序　e2,r,e1
	static bool cmp_tail(const Triple &a, const Triple &b) {
		return (a.t < b.t)||(a.t == b.t && a.r < b.r)||(a.t == b.t && a.r == b.r && a.h < b.h);
	}
        // 优先比较triple a,b 的e1 ——>e1小 True 比较顺序　e1,e2,r
	static bool cmp_rel(const Triple &a, const Triple &b) {
		return (a.h < b.h)||(a.h == b.h && a.t < b.t)||(a.h == b.h && a.t == b.t && a.r < b.r);
	}
        // 优先比较triple a,b 的r  ——>e1小 True 比较顺序　r,e1,e2
	static bool cmp_rel2(const Triple &a, const Triple &b) {
		return (a.r < b.r)||(a.r == b.r && a.h < b.h)||(a.r == b.r && a.h == b.h && a.t < b.t);
	}

};

#endif
