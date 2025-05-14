#ifndef SUPERPIXEL_HIERARCHY_MEX_HPP
#define SUPERPIXEL_HIERARCHY_MEX_HPP

#include <algorithm>
#include <iostream>

#define max(a,b) (((a) > (b)) ? (a) : (b))
#define min(a,b) (((a) < (b)) ? (a) : (b))

template<class T>
void radixSortLSD(int *I, int *Itemp, T *J, T *Jtemp, int length)
{
	int *beginI  = I,     *endI  =     I+length;
	int *beginIt = Itemp, *endIt = Itemp+length;
	T   *beginJ  = J,     *endJ  =     J+length;
	T   *beginJt = Jtemp, *endJt = Jtemp+length;
	for (int shift = 0; shift < 32; shift += 8) 
	{
		size_t count[0x100] = {};
		for (int *p = beginI; p != endI; p++)
			count[(*p >> shift) & 0xFF]++;
		int *bucketI[0x100], *qI = beginIt;
		T   *bucketJ[0x100], *qJ = beginJt;
		for (int i = 0; i < 0x100; ++i)
		{
			bucketI[i] = qI; bucketJ[i] = qJ;
			qI += count[i]; qJ += count[i];
		}
		int *p = beginI;
		T   *q = beginJ;
		for (; p != endI; p++, q++)
		{
			int idx = (*p >> shift) & 0xFF;
			*bucketI[idx]++ = *p;
			*bucketJ[idx]++ = *q;
		}
		std::swap(beginI, beginIt);
		std::swap(beginJ, beginJt);
		std::swap(endI, endIt);
		std::swap(endJ, endJt);
	}
}

struct ArcBox
{
	int u, v;
	ArcBox *ulink, *vlink;
	unsigned char confidence;
	//unsigned char   ismin;
	//int length;
	//ArcBox() : u(0), v(0), ulink(0), vlink(0), confidence(0) {} //
};

class SuperpixelHierarchy
{

private:
	ArcBox *m_arcmem;
	ArcBox **m_arcptr;
	ArcBox **m_frtarc;
	ArcBox **m_minarc;
	ArcBox **m_arctmp;

	int*m_vex, *m_parent, *m_label, *m_size;
	int m_vexmax, m_arcmax, m_vexnum, m_arcnum;
	int m_h, m_w, m_connect, m_iter, m_iterSwitch;

	unsigned short *m_color;
	int *m_dist, *m_treeu, *m_treev, *m_dtemp;
	int m_treeSize, m_regionnum;

	int *m_temp;

public:
	SuperpixelHierarchy()
	{
		m_arcmem = NULL;
		m_arcptr = NULL;
		m_frtarc = NULL;
		m_minarc = NULL;
		m_arctmp = NULL;
		m_color  = NULL;
		m_temp   = NULL;
	}

	~SuperpixelHierarchy() { clean(); }

	void init(int h, int w, int connect = 4, int iterSwitch = 4)
	{

		m_h = h; m_w = w; m_connect = connect; m_iterSwitch = iterSwitch;
		m_vexmax = m_vexnum = h*w;
		m_arcmax = computeEdge(h,w,connect); m_arcnum = 0;
		m_arcmem = new ArcBox [m_arcmax];
		m_arcptr = new ArcBox*[m_arcmax];
		m_minarc = new ArcBox*[m_vexnum]; 
		m_arctmp = new ArcBox*[m_vexnum];
		m_frtarc = new ArcBox*[m_vexnum]; 
		memset(m_frtarc,0,sizeof(ArcBox*)*m_vexnum);
		
		m_color  = new unsigned short[m_vexnum*3];
		m_temp   = new int[8*m_vexnum];
		m_vex    = m_temp+0*m_vexnum; 
		m_size   = m_temp+1*m_vexnum; 
		m_parent = m_temp+2*m_vexnum; 
		m_label  = m_temp+3*m_vexnum;
		m_dtemp  = m_temp+4*m_vexnum;  
		m_dist   = m_temp+5*m_vexnum; 
		m_treeu  = m_temp+6*m_vexnum;  
		m_treev  = m_temp+7*m_vexnum; 

		for (int i=0; i<m_vexnum; ++i) { m_vex[i] = m_parent[i] = m_label[i] = i; m_size[i] = 1; }
		for (int i=0; i<m_arcmax; ++i) { m_arcptr[i] = &m_arcmem[i]; } 
		m_regionnum = m_vexnum; m_treeSize = 0;

	}

	void buildTree(unsigned char *img, unsigned char *edge = NULL)
	{
		createVexLab(img);
		buildGraph(edge);

		m_iter = 0;
		int maxDistColor = 0;
		while (m_vexnum > 1)
		{
			++m_iter;
			nearestNeighbor(maxDistColor);
			growRegion();
			merge();
		}

		//printf("max color difference: %d\n", maxDistColor);
	}

	bool insertArc(int u, int v, unsigned char confidence = 0)
	{
		//if (isNeighbor(i, j)) return false;
		//if (i > j) std::swap(i, j); // let i be the small vertex

		ArcBox *arc  = m_arcptr[m_arcnum++];
		arc->u = u;
		arc->v = v;
		arc->ulink = m_frtarc[u];
		arc->vlink = m_frtarc[v];
		arc->confidence = confidence;
		m_frtarc[u] = m_frtarc[v] = arc;
		return true;
	}

	ArcBox *findArc(ArcBox *arc, int u, int v)
	{
		while (arc)
		{
			if (arc->v == v)
				return arc;
			else if (arc->v == u)
				arc = arc->vlink;
			else
				arc = arc->ulink;
		}
		return NULL;
	}

	void nearestNeighbor(int &maxDistColor)
	{

		for (int i=0; i<m_vexnum; ++i) m_dist[i] = INT_MAX;
		for (int n=0; n<m_arcnum; ++n)
		{
			ArcBox *arc  = m_arcptr[n];
			int u = arc->u;
			int v = arc->v;
			int lu = m_label[u];
			int lv = m_label[v];
			unsigned short *uc = m_color+3*u;
			unsigned short *vc = m_color+3*v;
			int dc[3];
			dc[0] = uc[0]-vc[0];
			dc[1] = uc[1]-vc[1];
			dc[2] = uc[2]-vc[2];

			// [1] color term
			int distColor = (abs(dc[0]) + abs(dc[1]) + abs(dc[2])) / 3;

			// [2] boundary term
			int distEdge = arc->confidence;

			int dist = distColor;
			if (m_iter > m_iterSwitch) dist *= distEdge;

			if (dist < m_dist[lu]) { m_dist[lu] = dist; m_minarc[lu] = arc; }
			if (dist < m_dist[lv]) { m_dist[lv] = dist; m_minarc[lv] = arc; }
		}

	}

	void growRegion()
	{

		// radix sort
		radixSortLSD(m_dist,m_dtemp,m_minarc,m_arctmp,m_vexnum);
				
		for (int i=0; i<m_vexnum; ++i)
		{
			ArcBox *arc = m_minarc[i];
			//if (arc->ismin)
				//continue;

			//arc->ismin = 1;
			int pu = findset(arc->u);
			int pv = findset(arc->v);
			if (pu == pv) continue;

			m_treeu[m_treeSize]   = pu;
			m_treev[m_treeSize++] = pv;
			if (pu < pv) m_parent[pv] = pu;
			else         m_parent[pu] = pv;
		}
		
		// update parent and label
		m_regionnum = 0;
		for (int i=0; i<m_vexnum; ++i)
		{
			int u = m_vex[i];
			int pu = findset(u);

			if (pu == u)
			{
				m_label[u] = m_regionnum;
				// create a new component
				m_vex[m_regionnum] = u;
				m_frtarc[m_regionnum] = NULL;
				++m_regionnum;
			}
			else
			{
				m_label[u] = m_label[pu];
				// update color
				int s1 = m_size[u]; int s2 = m_size[pu]; int s = s1+s2;
				m_size[pu] = s;
				m_color[3*pu+0] = (m_color[3*u+0]*s1 + m_color[3*pu+0]*s2) / s;
				m_color[3*pu+1] = (m_color[3*u+1]*s1 + m_color[3*pu+1]*s2) / s;
				m_color[3*pu+2] = (m_color[3*u+2]*s1 + m_color[3*pu+2]*s2) / s;
			}
		}
		m_vexnum = m_regionnum;

	}

	void merge()
	{

		// update connection
		int arcold = m_arcnum;
		m_arcnum = 0;
		for (int n=0; n<arcold; ++n)
		{
			ArcBox *arc = m_arcptr[n];
			int  u = arc->u;
			int  v = arc->v;
			int pu = m_parent[u];
			int pv = m_parent[v];
			if (pu == pv)
				continue;

			if (pu > pv) { std::swap(u,v); std::swap(pu,pv); }
			int lu = m_label[u]; int lv = m_label[v];
			ArcBox *&arcu = m_frtarc[lu];
			ArcBox *&arcv = m_frtarc[lv];
			ArcBox * find = findArc(arcu, pu, pv);
			if (find)
			{
				if (find->confidence > arc->confidence)
					find->confidence = arc->confidence;
			}
			else
			{
				//assert(arc->ismin == 0);
				arc->u = pu;
				arc->v = pv;
				arc->ulink = arcu;
				arc->vlink = arcv;
				m_arcptr[m_arcnum++] = arcu = arcv = arc;
			}
		}

	}

	int findset(int i)
	{
		int p = m_parent[i];
		if (i != p)
		{
			m_parent[i] = findset(p);
		}
		return m_parent[i];
	}

	int *getLabel()  { return m_label;     }
	int *getParent() { return m_parent;    }
	int *getTreeU()  { return m_treeu;     }
	int *getTreeV()  { return m_treev;     }
	int  getRegion() { return m_regionnum; }

	int *getLabel(int N)
	{
		if (N < 1 || N > m_vexmax)
		{
			printf("error");
			exit(1);
		}

		int end   = m_vexmax-N;
		int begin = m_vexmax-m_regionnum;
		if (m_regionnum < N)
		{
			for (int i=0; i<m_vexmax; ++i) m_parent[i] = i;
			begin = 0;
		}

		for (int i=begin; i<end; ++i)
		{
			int u  = m_treeu[i];
			int v  = m_treev[i];
			int pu = findset(u);
			int pv = findset(v);
			if (pu < pv)
				m_parent[pv] = pu;
			else
				m_parent[pu] = pv;
		}

		m_regionnum = 0;
		for (int i=0; i<m_vexmax; ++i)
		{
			int p = findset(i);
			if (i == p)
				m_label[i] = m_regionnum++;
			else
				m_label[i] = m_label[p];
		}

		return m_label;
	}

private:

	void clean()
	{
		delete [] m_arcptr;
		delete [] m_arcmem;
		delete [] m_frtarc;
		delete [] m_minarc;
		delete [] m_arctmp;
		delete [] m_color;
		delete [] m_temp;
	}

	int computeEdge(int h, int w, int connect)
	{
		if (h<=2 || w<=2)
			return 0;
		else if (connect == 4)
			return (h-1)*w + (w-1)*h;
		else
			return (h-1)*w + (w-1)*h + (h-1)*(w-1)*2;	
	}

	void createVexRGB(unsigned char *img)
	{
		unsigned char  *ptrr  = img;
		unsigned char  *ptrg  = ptrr+m_vexnum;
		unsigned char  *ptrb  = ptrg+m_vexnum;
		unsigned char  *ptre  = ptrg;
		unsigned short *ptrd  = m_color;
		while (ptrr != ptre)
		{
			*ptrd     = *ptrr++;
			*(ptrd+1) = *ptrg++;
			*(ptrd+2) = *ptrb++;
			ptrd += 3;
		}
	}

	void createVexLab(unsigned char *img)
	{
		// Lab color space
		const double Xr = 0.950456;	    //D65
		const double Yr = 1.0;
		const double Zr = 1.088754;

		double M[3][3] = { 0.412453, 0.357580, 0.180423,
			0.212671, 0.715160, 0.072169,
			0.019334, 0.119193, 0.950227};

		// Normalize for reference white point
		M[0][0] /= Xr; M[0][1] /= Xr; M[0][2] /= Xr;
		M[1][0] /= Yr; M[1][1] /= Yr; M[1][2] /= Yr;
		M[2][0] /= Zr; M[2][1] /= Zr; M[2][2] /= Zr;


		const double epsilon   = 1.0 / 3;
		const int    shift     = 20; // RGB->XYZ scale
		const int    halfShift = 1 << (shift-1);
		const int    scaleLC   = (int)(16 * (1 << shift));
		const int    scaleLT   = (int)(116);

		const int    LABXR    = (int)(M[0][0] * (1 << shift) + 0.5);
		const int    LABXG    = (int)(M[0][1] * (1 << shift) + 0.5);
		const int    LABXB    = (int)(M[0][2] * (1 << shift) + 0.5);
		const int    LABYR    = (int)(M[1][0] * (1 << shift) + 0.5);
		const int    LABYG    = (int)(M[1][1] * (1 << shift) + 0.5);
		const int    LABYB    = (int)(M[1][2] * (1 << shift) + 0.5);
		const int    LABZR    = (int)(M[2][0] * (1 << shift) + 0.5);
		const int    LABZG    = (int)(M[2][1] * (1 << shift) + 0.5);
		const int    LABZB    = (int)(M[2][2] * (1 << shift) + 0.5);

		// build Lab lockup table
		int table[4096]; // 4080 = 255*16
		const int thresh  = 36;
		for (int i=0; i<4096; ++i)
		{
			if (i > thresh)
				table[i] = (int)(pow(i / 4080.0, epsilon) * (1 << shift) + 0.5);
			else
				table[i] = (int)((29 * 29.0 * i / (6 * 6 * 3 * 4080) + 4.0 / 29) * (1 << shift) + 0.5 );
		}

		unsigned char *ptrr = img;
		unsigned char *ptrg = ptrr+m_vexnum;
		unsigned char *ptrb = ptrg+m_vexnum;
		unsigned short*ptrd = m_color;
		int R, G, B, X, Y, Z;
		for (int i=0; i<m_vexnum; ++i)
		{

			R = *ptrr++; G = *ptrg++; B = *ptrb++;  
			X = (B * LABXB + G * LABXG + R * LABXR + halfShift) >> (shift - 4); //RGB->XYZ x16
			Y = (B * LABYB + G * LABYG + R * LABYR + halfShift) >> (shift - 4);
			Z = (B * LABZB + G * LABZG + R * LABZR + halfShift) >> (shift - 4);
			X = table[X]; Y = table[Y]; Z = table[Z];

			*(ptrd+0) = (scaleLT * Y - scaleLC + halfShift) >> (shift - 4);         // L: (0~100) X 16
			*(ptrd+1) = ((500 * (X - Y) + halfShift) >> (shift - 4)) + (128 << 4);  // A: (0~255) X 16
			*(ptrd+2) = ((200 * (Y - Z) + halfShift) >> (shift - 4)) + (128 << 4);  // B: (0~255) X 16
			ptrd += 3;
		}
	}

	void createVexYCbCr(unsigned char *img)
	{
		const double M[3][3] = { 0.299,     0.587,     0.114,
			-0.168736, -0.331264,  0.5,
			0.5,       -0.418688, -0.081312};

		const int shift     = 20; // RGB->YCbCr scale
		const int halfShift = 1 << (shift-1);
		const int YCbCrYR  = (int)(M[0][0] * (1 << shift) + 0.5);
		const int YCbCrYG  = (int)(M[0][1] * (1 << shift) + 0.5);
		const int YCbCrYB  = (int)(M[0][2] * (1 << shift) + 0.5);
		const int YCbCrCbR = (int)(M[1][0] * (1 << shift) + 0.5);
		const int YCbCrCbG = (int)(M[1][1] * (1 << shift) + 0.5);
		const int YCbCrCbB = (int)(M[1][2] * (1 << shift) + 0.5);
		const int YCbCrCrR = (int)(M[2][0] * (1 << shift) + 0.5);
		const int YCbCrCrG = (int)(M[2][1] * (1 << shift) + 0.5);
		const int YCbCrCrB = (int)(M[2][2] * (1 << shift) + 0.5);

		unsigned char  *ptrr  = img;
		unsigned char  *ptrg  = ptrr+m_vexnum;
		unsigned char  *ptrb  = ptrg+m_vexnum;
		unsigned char  *ptre  = ptrg;
		unsigned short *ptrd  = m_color;
		int R,G,B;
		while (ptrr != ptre)
		{
			R = *ptrr++; G = *ptrg++; B = *ptrb++;
			*ptrd     = (unsigned short)(               (YCbCrYR  * R + YCbCrYG  * G + YCbCrYB  * B + halfShift) >> (shift-4));
			*(ptrd+1) = (unsigned short)( (128 << 4) + ((YCbCrCbR * R + YCbCrCbG * G + YCbCrCbB * B + halfShift) >> (shift-4)));
			*(ptrd+2) = (unsigned short)( (128 << 4) + ((YCbCrCrR * R + YCbCrCrG * G + YCbCrCrB * B + halfShift) >> (shift-4)));
			ptrd += 3;
		}

	}

	void buildGraph(unsigned char *edge)
	{
		// x direction
		for (int y=0; y<m_h; ++y)
		{
			int id0 = y;
			int idt = id0+m_h;
			for (int x=0; x<m_w-1; ++x)
			{
				unsigned char boundary = min(edge[id0], edge[idt]);
				//boundary = boundary + 1;
				boundary = max(boundary,1);
				insertArc(id0,idt,boundary);
				id0 = idt; idt += m_h;
			}
		}

		// y direction
		for (int x=0; x<m_w; ++x)
		{
			int id0 = x*m_h;
			int idt = id0+1;
			for (int y=0; y<m_h-1; ++y)
			{
				unsigned char boundary = min(edge[id0], edge[idt]);
				//boundary = boundary + 1;
				boundary = max(boundary,1);
				insertArc(id0,idt,boundary);
				id0 = idt++;
			}
		}

		if (m_connect != 4)
		{
			for (int y0=0; y0<m_h-1; ++y0)
			{
				int yt = y0+1;
				for (int x0=0; x0<m_w-1; ++x0)
				{
					int xt = x0+1;
					int id0 = x0*m_h+y0;
					int idt = xt*m_h+yt;
					unsigned char boundary = min(edge[id0], edge[idt]);
					//boundary = boundary + 1;
					boundary = max(boundary,1);
					insertArc(id0, idt, boundary);
					id0 = xt*m_h+y0;
					idt = x0*m_h+yt;
					boundary = min(edge[id0], edge[idt]);
					//boundary = boundary + 1;
					boundary = max(boundary,1);
					insertArc(id0, idt, boundary);
				}
			}
		}
	}

};

#endif