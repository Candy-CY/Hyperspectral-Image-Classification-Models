#include "mex.h"

int findset(int i, int *parent)
{
	int p = parent[i];
	if (i != p)
	{
		parent[i] = findset(p, parent);
	}
	return parent[i];
}

void getSuperpixels(int *parent, int *label, int *treeu, int *treev, int &nvertex, int &nregion, int &N)
{
	if (N < 1 || N > nvertex)
	{
		printf("error");
		exit(1);
	}

	int end   = nvertex-N;
	int begin = nvertex-nregion;
	if (nregion < N)
	{
		for (int i=0; i<nvertex; ++i) parent[i] = i;
		begin = 0;
	}

	for (int i=begin; i<end; ++i)
	{
		int u  = treeu[i];
		int v  = treev[i];
		int pu = findset(u,parent);
		int pv = findset(v,parent);
		if (pu < pv)
			parent[pv] = pu;
		else
			parent[pu] = pv;
	}

	nregion = 0;
	for (int i=0; i<nvertex; ++i)
	{
		int p = findset(i,parent);
		if (i == p)
			label[i] = nregion++;
		else
			label[i] = label[p];
	}
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[])
{
	if (nrhs != 2)
		mexErrMsgTxt("usage: GetSuperpixels(model, nsuperpixels)");

	int superpixels = mxGetScalar(prhs[1]);
	int *parent  = (int *)mxGetData(mxGetFieldByNumber(prhs[0],0,0));
	int *label   = (int *)mxGetData(mxGetFieldByNumber(prhs[0],0,1));
	int *treeu   = (int *)mxGetData(mxGetFieldByNumber(prhs[0],0,2));
	int *treev   = (int *)mxGetData(mxGetFieldByNumber(prhs[0],0,3));
	int *nvertex = (int *)mxGetData(mxGetFieldByNumber(prhs[0],0,4));
	int *nregion = (int *)mxGetData(mxGetFieldByNumber(prhs[0],0,5));
	getSuperpixels(parent,label,treeu,treev,*nvertex,*nregion,superpixels);
}