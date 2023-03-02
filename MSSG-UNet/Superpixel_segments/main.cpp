#include "SuperpixelHierarchyMex.hpp"
#include "basic.h"
#include "ppm.h"

void transposeAndSplit(unsigned char *out, unsigned char *in, int h, int w)
{
	int hw = h*w, hw2 = 2*h*w;
	for (int y=0; y<h; ++y) for (int x=0; x<w; ++x)
	{
		out[x*h+y] = in[3*(y*w+x)+0];
		out[hw+x*h+y] = in[3*(y*w+x)+1];
		out[hw2+x*h+y] = in[3*(y*w+x)+2];
	}
}

template<class T>
void transpose(T *out, T *in, int h, int w)
{
	for (int y=0; y<h; ++y) for (int x=0; x<w; ++x) out[x*h+y] = in[y*w+x];
}

template<class T>
void transposeT(T *out, T *in, int h, int w)
{
	for (int x=0; x<w; ++x) for (int y=0; y<h; ++y) out[y*w+x] = in[x*h+y];
}

void benchmark(char *fin_img, char *fin_edge, char *fout, int nr_superpixel, int nr_neighbor, int iterSwitch)
{
	char str[999]; int str_len=999; char ext_img[] = "*.ppm"; char ext_edge[] = "*.pgm";
	qx_create_director(fout);
	int nr_image=get_nr_file(fin_img,ext_img);

	char**filename_img  = qx_allocc(nr_image,str_len);
	char**filename_edge = qx_allocc(nr_image,str_len);
	get_files(filename_img, fin_img, ext_img, str_len);
	get_files(filename_edge,fin_edge,ext_edge,str_len);
	double total_time(0);
	for(int i=0;i<nr_image;i++)
	{
		sprintf_s(str,str_len,"%s/%s",fin_img,filename_img[i]);
		int h, w;
		qx_image_size(str, h, w);
		unsigned char *img   = new unsigned char[3*h*w];
		unsigned char *imgt  = new unsigned char[3*h*w];
		unsigned char *color = new unsigned char[3*h*w];
		unsigned char *edge  = new unsigned char[h*w];
		unsigned char *edget = new unsigned char[h*w];
		int *labelt = new int[h*w];
		qx_loadimage(str, img, h, w);
		sprintf_s(str,str_len,"%s/%s",fin_edge,filename_edge[i]);
		qx_loadimage(str, edge, h, w);

		transposeAndSplit(imgt,img,h,w);
		transpose(edget,edge,h,w);
		//image_display(edge,h,w);
		SuperpixelHierarchy seg;
		seg.init(h,w,nr_neighbor,iterSwitch);
		seg.buildTree(imgt, edget);
		int *label = seg.getLabel(nr_superpixel);
		transposeT(labelt,label,h,w);
		label = labelt;
		//seg.draw_contours(0, 0, 0, 0);
		//seg.getMeanColor(img,color);
		//image_display(labelt, h, w);

		//char strr[1000];
		//sprintf(strr, "%s", filename[i]);
		//int len = strlen(strr);
		//strr[len-4] = 0;
		//sprintf_s(str,str_len,"%s/%s.ppm",fout,strr);
		//qx_saveimage(str, color, h, w, 3);

		char strr[1000];
		sprintf(strr, "%s", filename_img[i]);
		int len = strlen(strr);
		strr[len-4] = 0;
		sprintf_s(str,str_len,"%s/%s.txt",fout,strr);
		FILE * fileout; int y,x;
		fopen_s(&fileout,str,"w");
		for(y=0;y<h;y++)
		{	
			for(x=0;x<w;x++)
			{
				fprintf(fileout,"%i ",*label);
				++label;
			}
			fprintf(fileout,"\n");	
		}
		fclose(fileout);

		printf("[%d]/[%d]\n",i+1,nr_image);
		delete [] img; delete [] imgt;
		delete [] color;
		delete [] edge; delete [] edget;
		delete [] labelt;
	}

	double avg_time = total_time / nr_image;
	printf("Running time: total[%5.5f], average[%5.5f].\n", total_time, avg_time);
}

void main(int argc, char **argv)
{
	//test();
	//test_edge();

	int nr_superpixel = 600;
	int nr_neighbor = 4;
	//char fin_img[]  = "D:/Superpixel/BSR/BSDS500/images/test_ppm";
	char fin_img[]  = "D:/Filter/Gaussian/BSDS500_ppm";
	//char fin_img[]  = "E:/Filters/RecursiveBilateralFilter/Matlab_RBF/BSDS500_edge_ppm";
	//char fin_edge[] = "E:/EdgesDetection/OrientedEdge/oef-master/BSDS500_BL2";
	char fin_edge[] = "D:/EdgesDetection/StructuredEdges/edges/BSDS500_BL4";
	//char fin_edge[] = "E:/EdgesDetection/canny/BSDS500_pgm";
	//char fin_edge[] = "D:/EdgesDetection/FastAccurate/dist/BSDS500";
	//char fin_edge[] = "D:/EdgesDetection/NCC_Edge/BSDS500";
	//char fin_mask[] = "E:/EdgesDetection/Canny/BSDS500";
	char fout[]     = "D:/Superpixel/sh_hist_600";

	//benchmark(fin_img, fout, nr_superpixel, nr_neighbor);
	benchmark(fin_img, fin_edge, fout, nr_superpixel, nr_neighbor, 0);
	getchar();
}