#include "mex.h"
#include "SuperpixelHierarchyMex.hpp"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	// model = superpixelHirarchyMex(image, edge, connect, iter);
	if (nrhs < 2 || nrhs > 4)
		mexErrMsgTxt("usage: model = superpixelHirarchyMex(image, edge, [connect], [iter])");
	if (mxGetClassID(prhs[0]) != mxUINT8_CLASS)
		mexErrMsgTxt("Input image must be uint8.");
	if (mxGetClassID(prhs[1]) != mxUINT8_CLASS)
		mexErrMsgTxt("Input edge map must be uint8.");

	unsigned char *image = (unsigned char *)mxGetData(prhs[0]);
	unsigned char *edge  = (unsigned char *)mxGetData(prhs[1]);
	int *dims = (int *)mxGetDimensions(prhs[0]);
	int h = dims[0]; int w = dims[1];
	int c = mxGetNumberOfElements(prhs[0]) / (h*w);
	if (c != 3)
		mexErrMsgTxt("Input must be 24 bit color image.");
	int connect = 4; int iterSwitch = 4;
	if (nrhs > 2)
		connect = mxGetScalar(prhs[2]);
	if (nrhs > 3)
		iterSwitch = mxGetScalar(prhs[3]);

	SuperpixelHierarchy SH;
	SH.init(h,w,connect,iterSwitch);
	SH.buildTree(image,edge);

	const char *fieldnames[] = {"parent","label","treeu","treev","nvertex","nregion"};
	mxArray *parent   = mxCreateNumericMatrix(h,w,mxINT32_CLASS,0);
	mxArray *label    = mxCreateNumericMatrix(h,w,mxINT32_CLASS,0);
	mxArray *treeu    = mxCreateNumericMatrix(h*w-1,1,mxINT32_CLASS,0);
	mxArray *treev    = mxCreateNumericMatrix(h*w-1,1,mxINT32_CLASS,0);
	mxArray *nvertex  = mxCreateNumericMatrix(1,1,mxINT32_CLASS,0);
	mxArray *nregion  = mxCreateNumericMatrix(1,1,mxINT32_CLASS,0);
	memcpy(mxGetData(parent), SH.getParent(), sizeof(int)*h*w);
	memcpy(mxGetData(label),  SH.getLabel(),  sizeof(int)*h*w);
	memcpy(mxGetData(treeu),  SH.getTreeU(),  sizeof(int)*(h*w-1));
	memcpy(mxGetData(treev),  SH.getTreeV(),  sizeof(int)*(h*w-1));
	*(int*)mxGetData(nvertex) = h*w; *(int*)mxGetData(nregion) = SH.getRegion();
	plhs[0] = mxCreateStructMatrix(1,1,6,fieldnames);
	mxSetFieldByNumber(plhs[0],0,0,parent);
	mxSetFieldByNumber(plhs[0],0,1,label);
	mxSetFieldByNumber(plhs[0],0,2,treeu);
	mxSetFieldByNumber(plhs[0],0,3,treev);
	mxSetFieldByNumber(plhs[0],0,4,nvertex);
	mxSetFieldByNumber(plhs[0],0,5,nregion);
}