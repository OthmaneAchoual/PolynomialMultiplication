#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <curand_kernel.h>

#include <fstream>
#include <iostream>

__device__ __host__ unsigned int bitreverse(unsigned int number) {
	number = ((0xffff0000 & number) >> 16) | ((0x0000ffff & number) << 16);
	number = ((0xff00ff00 & number) >> 8)  | ((0x00ff00ff & number) << 8);
	number = ((0xf0f0f0f0 & number) >> 4)  | ((0x0f0f0f0f & number) << 4);
	number = ((0xcccccccc & number) >> 2)  | ((0x33333333 & number) << 2);
	number = ((0xaaaaaaaa & number) >> 1)  | ((0x55555555 & number) << 1);
	return number;
}

__device__ __host__ int fastexp(int b, int e, int p)
{
	int r = 1;
	while (e > 0)
	{
		if (e & 0x1)
			r = ((long)r * b) % p;
		e >>= 1;
		b = ((long)b * b) % p;
	}
	return r;
}

__device__ __host__ unsigned int getIndex(int m, int j)
{
	unsigned int m1 = (m >> j) << (j + 1);
	unsigned int m2 = ((1 << j) - 1) & m;

	return (m1 | m2);
}

__device__ __host__ unsigned int getShift(unsigned int index, int j, int n)
{
	unsigned int r = index >> (j + 1);
	r     = bitreverse(r);
	r   >>= 32 - ((n >> 1) - j);
	r   <<= j;

	return r;
}

__global__ void rand_device_api_kernel(curandState *states, int *out, int n, int p)
{
	int gtid = blockIdx.x * blockDim.x + threadIdx.x;

	if (gtid < n)
	{
		curandState *state = states + gtid;
		curand_init(1234UL, gtid, 0, state);

		double rand   = curand_uniform_double(state);
		out[gtid]     = (int)floor(p*rand);
		out[gtid + n] = 0;
	}
}

__global__ void montgomeryKernel(int *a, int *b, long *c, int n, int p, int thres)
{
		long temp, temp0;

		int tid    = threadIdx.x;
		int inc    = thres;
		long *pC   = c + (blockIdx.x * n);
		//int index;

		for(int i = tid; i < n; i += blockDim.x)
			pC[i] = 0;

		__syncthreads();

			for (int i = 0; i < n; i++)
			{
				for(int j = tid; j < n; j += blockDim.x)
				{
					if(j == 0)
						temp0 = pC[j] + ((long)a[j] * b[i]);
					else
						temp  = pC[j] + ((long)a[j] * b[i]);

					if (i == thres)
					{
						temp0 %= p;
						temp  %= p;
					}

					if(j != 0)
						pC[j-1] = temp;

					__syncthreads();
				}

				if (tid == 0)
					pC[n-1] = -temp0;

				if(i == thres)
					thres += inc;

				__syncthreads();
			}

			for(int i = tid; i < n; i += blockDim.x)
				pC[i] = ((-pC[i] % p) + p) % p;
}

//__global__ void nttKernel(int *a, int *b, int *c, int n, int *wn, int *wni, int p, int ni)
//{
//	extern __shared__ int local[];
//
//	int *locala = local;
//	int *localb = local + n;
//
//	int tid    = threadIdx.x;
//	int lgn    = (int)roundf(log2f((float)n));
//
//	// Copy elements in reverse bit order to local memory
//	locala[2*tid]   = a[bitreverse(2*tid)   >> (32 - lgn)];
//	locala[2*tid+1] = a[bitreverse(2*tid+1) >> (32 - lgn)];
//
//	localb[2*tid]   = b[bitreverse(2*tid)   >> (32 - lgn)];
//	localb[2*tid+1] = b[bitreverse(2*tid+1) >> (32 - lgn)];
//
//	__syncthreads();
//
//	int quotient = tid; // quotient = tid / stride
//
//	// Forward NTT
//	for(int s = 1; s <= lgn; s++)
//	{
//		// Set up what we need for this iteration
//		int m        = 1 << s;
//		int stride   = m >> 1;
//
//		int factor   = tid - (quotient << (s-1));
//		int t_tid    = (quotient << s) + factor;
//
//		int w          = wn[(lgn - s + 1) * (n >> 1) + factor];
//		// Butterfly operation
//		long t                  = (long)w * locala[t_tid + stride];
//		int u                   = locala[t_tid];
//		locala[t_tid]           = (u + t) % p;
//		locala[t_tid + stride]  = (u - t) % p;
//
//		t                      = (long)w * localb[t_tid + stride];
//		u                      = localb[t_tid];
//		localb[t_tid]          = (u + t) % p;
//		localb[t_tid + stride] = (u - t) % p;
//
//		quotient >>= 1;
//
//		__syncthreads();
//	}
//
//	localb[2*tid]   = ((long)localb[2*tid] * locala[2*tid]) % p;
//	localb[2*tid+1] = ((long)localb[2*tid+1] * locala[2*tid+1]) % p;
//
//	__syncthreads();
//
//	locala[2*tid]   = localb[bitreverse(2*tid)   >> (32 - lgn)];
//	locala[2*tid+1] = localb[bitreverse(2*tid+1) >> (32 - lgn)];
//
//	__syncthreads();
//
//	quotient = tid;
//
//	// Backward NTT
//	for(int s = 1; s <= lgn; s++)
//	{
//		int m        = 1 << s;
//		int stride   = m >> 1;
//		int factor   = tid - (quotient << (s-1));
//		int t_tid    = (quotient << s) + factor;
//		int w        = wni[(lgn - s + 1) * (n >> 1) + factor];
//
//		long t                 = (long)w * locala[t_tid + stride];
//		int u                  = locala[t_tid];
//		locala[t_tid]          = (u + t) % p;
//		locala[t_tid + stride] = (u - t) % p;
//
//		quotient >>= 1;
//
//		__syncthreads();
//	}
//
//	// write results to buffer
//	if(blockIdx.x == 0)
//	{
////		c[2*tid]   = locala[2*tid];
////		c[2*tid+1] = locala[2*tid+1];
//		locala[2*tid]   = (locala[2*tid] * ni) % p;
//		locala[2*tid+1] = (locala[2*tid+1] * ni) % p;
//
//		__syncthreads();
//
//		c[tid] = (((locala[tid] - locala[tid + (n/2)]) % p) + p) % p;
//	}
//}


__global__ void nttKernel(int *a, int *b, int *c, int n, int *wn, int *wni, int p, int ni)
{
	int tid    = threadIdx.x;
	int bid    = blockIdx.x;
	int lgn    = (int)roundf(log2f((float)n));

	int *pC1   = c   + (bid * n * 2);
	int *pC2   = pC1 + n;

	// Copy elements in reverse bit order to scratchpad memory
	for (int i = tid; i < n/2; i += blockDim.x)
	{
		pC1[2*i]   = b[bitreverse(2*i)   >> (32 - lgn)];
		pC1[2*i+1] = b[bitreverse(2*i+1) >> (32 - lgn)];
	}
	__syncthreads();

	for (int i = tid; i < n/2; i += blockDim.x)
	{
		pC2[2*i]   = a[bitreverse(2*i)   >> (32 - lgn)];
		pC2[2*i+1] = a[bitreverse(2*i+1) >> (32 - lgn)];
	}

	int *locala = pC2;
	int *localb = pC1;

	__syncthreads();

	// Forward NTT
	for(int s = 1; s <= lgn; s++)
	{
		// Set up what we need for this iteration
		for (int i = tid; i < n/2; i += blockDim.x)
		{
			int m        = 1 << s;
			int stride   = m >> 1;
			int quotient = i / stride;

			int factor   = i - (quotient << (s-1));
			int t_tid    = (quotient << s) + factor;

			int w          = wn[(lgn - s + 1) * (n >> 1) + factor];
			// Butterfly operation
			long t                  = ((long)w * locala[t_tid + stride]) % p;
			long u                  = locala[t_tid];
			locala[t_tid]           = (u + t) % p;
			locala[t_tid + stride]  = (((u - t) % p) + p) % p;

			t                      = ((long)w * localb[t_tid + stride]) % p;
			u                      = localb[t_tid];
			localb[t_tid]          = (u + t) % p;
			localb[t_tid + stride] = (((u - t) % p) + p) % p;
		}
		__syncthreads();
	}

	for(int i=tid; i < n/2; i+=blockDim.x)
	{
		localb[2*i]   = ((long)localb[2*i] * locala[2*i]) % p;
		localb[2*i+1] = ((long)localb[2*i+1] * locala[2*i+1]) % p;
	}
	__syncthreads();

	for(int i=tid; i < n/2; i+=blockDim.x)
	{
		locala[2*i]   = localb[bitreverse(2*i)   >> (32 - lgn)];
		locala[2*i+1] = localb[bitreverse(2*i+1) >> (32 - lgn)];
	}
	__syncthreads();

	// Backward NTT
	for(int s = 1; s <= lgn; s++)
	{
		for(int i=tid; i < n/2; i+=blockDim.x)
		{
			int m        = 1 << s;
			int stride   = m >> 1;
			int quotient = i / stride;
			int factor   = i - (quotient << (s-1));
			int t_tid    = (quotient << s) + factor;
			int w        = wni[(lgn - s + 1) * (n >> 1) + factor];

			long t                 = ((long)w * locala[t_tid + stride]) % p;
			long u                 = locala[t_tid];
			locala[t_tid]          = (u + t) % p;
			locala[t_tid + stride] = (((u - t) % p) + p) % p;
		}
		__syncthreads();
	}

	for(int i=tid; i < n/2; i+= blockDim.x)
	{
		locala[2*i]   = ((long)locala[2*i] * ni) % p;
		locala[2*i+1] = ((long)locala[2*i+1] * ni) % p;
	}
	__syncthreads();

	for(int i=tid; i < n/2; i+=blockDim.x)
	{
		pC1[i] = (((locala[i] - locala[i + (n/2)]) % p) + p) % p;
	}
}

//__global__ void nussKernel(int *rx, int *ry, int *ns, int lgn, int *wx, int *wy, int scratchSize, int *z)
//{
//	int bid = blockIdx.x;
//	int tid = threadIdx.x;
//
//	// each block has its reserved scratchpad space
//	int *brx = bid * scratchSize + rx;
//	int *bry = bid * scratchSize + ry;
//	int *bwx = bid * scratchSize + wx;
//	int *bwy = bid * scratchSize + wy;
//
//	int numTrans        = 1;
//	int numTransPerPass;
//	int numPasses;
//
//	for (int l = 0; l < lgn; l++)
//	{
//		int n         = ns[l];
//		int TransSize = 1 << n;
//		int m         = 1 << (n >> 1);
//		int r         = (n & 0x1) ? m << 1 : m;
//
//		numTransPerPass = blockDim.x / TransSize;
//		numPasses       = numTrans   / numTransPerPass;
//
//		int ptid = tid; // thread id during the different passes
//		for (int k = 0; k < numPasses; k++)
//		{
//			int lbid = ptid / TransSize;
//			int ltid = ptid - (lbid * TransSize);
//
//			int *lrx = lbid * TransSize + brx;
//			int *lry = lbid * TransSize + bry;
//
//			int *lwx = lbid * (m << 1) * r + bwx;
//			int *lwy = lbid * (m << 1) * r + bwy;
//
//			int j = ltid / m;
//			int i = ltid - (j * m);
//
//			// Initialize
//			lwx[i * r + j] = lwx[(i + m)*r + j] = lrx[ltid];
//			lwy[i * r + j] = lwy[(i + m)*r + j] = lry[ltid];
//
//			int lm = ltid / r;
//			int lr = ltid - (lm * r);
//
//			// Transform
//			for (int j = (n >> 1) - 1; j >= 0; j--)
//			{
//				int index  = getIndex(lm, j);
//				int shift  = getShift(index, j, n);
//				shift     *= (r == m) ? 1 : 2;
//				int sindex = (lr >= shift) ? (lr - shift) : (lr - shift + r);
//
//				int tx                            = lwx[index * r + lr];
//				int ux                            = (lr >= shift) ? lwx[(index + (1 << j)) * r + sindex] : -lwx[(index + (1 << j)) * r + sindex];
//
//				int ty                            = lwy[index * r + lr];
//				int uy                            = (lr >= shift) ? lwy[(index + (1 << j)) * r + sindex] : -lwy[(index + (1 << j)) * r + sindex];
//
//				__syncthreads();
//
//				lwx[index * r + lr]              = tx + ux;
//				lwx[(index + (1 << j)) * r + lr] = tx - ux;
//
//				lwy[index * r + lr]              = ty + uy;
//				lwy[(index + (1 << j)) * r + lr] = ty - uy;
//
//				__syncthreads();
//			}
//
//			ptid += blockDim.x;
//		} // End of passes
//
//		// update the state
//		numTrans *= (m << 1);
//
//		int *temp = brx; brx = bwx; bwx = temp;
//		temp      = bry; bry = bwy; bwy = temp;
//
//		__syncthreads();
//	} // End of this stage
//
//	// Do all the convolutions of size 2
//	for (int i = tid; i < numTrans; i += blockDim.x)
//	{
//		// Determine on which data to operate
//		int *lrx = (i << 1) + brx;
//		int *lry = (i << 1) + bry;
//		int *lwx = (i << 1) + bwx;
//
//		// Operate
//		int x0, x1, y0, y1;
//		x0     = lrx[0];	x1 = lrx[1];	y0 = lry[0];	y1 = lry[1];
//		int t  = x0 * (y0 + y1);
//		lwx[0] = t - (x0 + x1) * y1;
//		lwx[1] = t + (x1 - x0) * y0;
//	}
//
//	// Set read and write segments for the last phase
//	int *brz = bwx;
//	int *bwz = bwy;
//
//	// Start last phase
//	for (int l = lgn-1; l >= 0; l--)
//	{
//		int n        = ns[l];
//		int m        = 1 << (n >> 1);
//		int r        = (n & 0x1) ? m << 1 : m;
//		int blocking = r * m; // how many threads per reconstruction
//		int grouping = m << 1; // how many polys needed for reconstruction
//
//		numTransPerPass = blockDim.x / blocking;
//		numPasses       = (numTrans  / grouping) / numTransPerPass;
//
//		int ptid = tid;
//		for (int k = 0; k < numPasses; k++)
//		{
//			int lbid = ptid / blocking;
//			int ltid = ptid - (lbid * blocking);
//			//where to read from
//			int *lrz = grouping * r * lbid + brz;
//
//			//where to write to
//			int *lwz = lbid * blocking + bwz; // offset is current lbid * size of trans at next level
//
//			int lm = ltid / r;
//			int lr = ltid - (lm * r);
//
//			// Untransform
//			for (int j = 0; j <= (n >> 1); j++)
//			{
//				int index = getIndex(lm, j);
//				int shift = getShift(index, j, n);
//				shift *= (r == m) ? 1 : 2;
//				//int sindex = (lr < (r - shift)) ? (lr + shift) : (lr + shift - r);
//				int sindex;
//				int tp, up;
//
//				int t = lrz[index * r + lr];
//				int u = lrz[(index + (1 << j)) * r + lr];
//
//				if (lr < (r - shift))
//				{
//					sindex = lr + shift;
//					tp = lrz[index * r + sindex];
//					up = lrz[(index + (1 << j)) * r + sindex];
//				}
//				else
//				{
//					sindex = lr + shift - r;
//					tp = -lrz[index * r + sindex];
//					up = -lrz[(index + (1 << j)) * r + sindex];
//				}
//
//				__syncthreads();
//
//				lrz[index * r + lr] = (t + u) / 2;
//				lrz[(index + (1 << j)) * r + lr] = (tp - up) / 2;
//
//				__syncthreads();
//			}
//
//			// Repack
//			int j = ltid / m;
//			int i = ltid - (j * m);
//
//			lwz[ltid] = (j == 0) ? (lrz[i * r] - lrz[(m + i) * r + (r - 1)]) : (lrz[i * r + j] + lrz[(m + i) * r + (j - 1)]);
//
//			ptid += blockDim.x;
//		}
//
//		int *temp = brz; brz = bwz; bwz = temp;
//		numTrans /= grouping;
//
//		__syncthreads();
//	}
//
//	z[tid] = brz[tid];
//}

__global__ void nussbaumerKernel(int *rx, int *ry, int *ns, int lgn, int *wx, int *wy, int scratchSize, int *z, int p, long inv2)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;

	// each block has its reserved scratchpad space
	int *brx = bid * scratchSize  + rx;
	int *bry = bid * scratchSize  + ry;
	int *bwx = bid * scratchSize  + wx;
	int *bwy = bid * scratchSize  + wy;
	int *bz  = bid * (1 << ns[0]) + z;

	int numTrans = 1;
	int numTransPerPass;
	int numPasses;

	for (int l = 0; l < lgn; l++)
	{
		int n = ns[l];
		int TransSize = 1 << n;
		int m = 1 << (n >> 1);
		int r = (n & 0x1) ? m << 1 : m;

		numTransPerPass = blockDim.x / r;
		numPasses = numTrans / numTransPerPass;

		int ptid = tid; // thread id during the different passes
		for (int k = 0; k < numPasses; k++)
		{
			int lbid = ptid / r;
			int ltid = ptid - (lbid * r);

			int *lrx = lbid * TransSize + brx;
			int *lry = lbid * TransSize + bry;

			int *lwx = lbid * (m << 1) * r + bwx;
			int *lwy = lbid * (m << 1) * r + bwy;

			for (int o = ltid; o < (1 << n); o += r)
			{
				int j = o / m;
				int i = o - (j * m);
				// Initialize
				lwx[i * r + j] = lwx[(i + m)*r + j] = lrx[o];
				lwy[i * r + j] = lwy[(i + m)*r + j] = lry[o];
			}

			// Transform
			for (int j = (n >> 1) - 1; j >= 0; j--)
			{
				for (int lm = 0; lm < m; lm++)
				{
					int index  = getIndex(lm, j);
					int shift  = getShift(index, j, n);
					shift     *= (r == m) ? 1 : 2;
					int sindex = (ltid >= shift) ? (ltid - shift) : (ltid - shift + r);

					int tx = lwx[index * r + ltid];
					int ux = (ltid >= shift) ? lwx[(index + (1 << j)) * r + sindex] : -lwx[(index + (1 << j)) * r + sindex];

					int ty = lwy[index * r + ltid];
					int uy = (ltid >= shift) ? lwy[(index + (1 << j)) * r + sindex] : -lwy[(index + (1 << j)) * r + sindex];

					__syncthreads();

					lwx[index * r + ltid]              = ((tx + ux) % p);
					lwx[(index + (1 << j)) * r + ltid] = (((tx - ux) % p) + p) % p;

					lwy[index * r + ltid]              = ((ty + uy) % p);
					lwy[(index + (1 << j)) * r + ltid] = (((ty - uy) % p) + p) % p;

					__syncthreads();
				}

			}

			ptid += blockDim.x;
		} // End of passes
		// update the state
		numTrans *= (m << 1);
		//if(tid == 0) printf("%d\n", numTrans);

		int *temp = brx; brx = bwx; bwx = temp;
		temp = bry; bry = bwy; bwy = temp;

		__syncthreads();
	} // End of this stage

	// Determine how many passes to operate on the whole array
	for (int i = tid; i < numTrans; i += blockDim.x)
	{
		// Determine on which data to operate

		int *lrx = (i << 1) + brx;
		int *lry = (i << 1) + bry;
		int *lwx = (i << 1) + bwx;
		//int *lwy = i * TransSize + bwy;

		// Operate
		int x0, x1, y0, y1;
		x0 = lrx[0];	x1 = lrx[1];	y0 = lry[0];	y1 = lry[1];
		long t = (long)x0 * (y0 + y1);
		lwx[0] = (((t - (x0 + x1) * (long)y1) % p) + p) % p;
		lwx[1] = ((t + (x1 - x0) * (long)y0) % p);

		//ptid += blockDim.x;
	}

	int *brz = bwx;
	int *bwz = bwy;

	// Start last phase
	for (int l = lgn - 1; l >= 0; l--)
	{
		int n = ns[l];
		int m = 1 << (n >> 1);
		int r = (n & 0x1) ? m << 1 : m;
		int blocking = r; // how many threads per reconstruction
		int grouping = m << 1; // how many polys needed for reconstruction

		numTransPerPass = blockDim.x / blocking;
		numPasses = (numTrans / grouping) / numTransPerPass;

		int ptid = tid;
		for (int k = 0; k < numPasses; k++)
		{
			int lbid = ptid / blocking;
			int ltid = ptid - (lbid * blocking);
			//where to read from
			int *lrz = grouping * r * lbid + brz;

			//where to write to
			int *lwz = lbid * (r * m) + bwz; // offset is current lbid * size of trans at next level

			// Untransform
			for (int j = 0; j <= (n >> 1); j++)
			{
				for (int lm = 0; lm < m; lm++)
				{
					int index = getIndex(lm, j);
					int shift = getShift(index, j, n);
					shift    *= (r == m) ? 1 : 2;

					int sindex;
					int tp, up;

					int t = lrz[index * r + ltid];
					int u = lrz[(index + (1 << j)) * r + ltid];

					if (ltid < (r - shift))
					{
						sindex = ltid + shift;
						tp = lrz[index * r + sindex];
						up = lrz[(index + (1 << j)) * r + sindex];
					}
					else
					{
						sindex = ltid + shift - r;
						tp = -lrz[index * r + sindex];
						up = -lrz[(index + (1 << j)) * r + sindex];
					}

					__syncthreads();

					lrz[index * r + ltid]              = (((t + u) * (long)inv2) % p);
					lrz[(index + (1 << j)) * r + ltid] = ((((tp - up) * (long)inv2) % p) + p) % p;

					__syncthreads();
				}
			}
			// Repack
			for (int o = ltid; o < (1 << n); o += r)
			{
				int j = o / m;
				int i = o - (j * m);

				lwz[o] = (j == 0) ? (((lrz[i * r] - lrz[(m + i) * r + (r - 1)]) % p) + p) % p : ((lrz[i * r + j] + lrz[(m + i) * r + (j - 1)]) % p);
			}
			ptid += blockDim.x;
		}

		int *temp = brz; brz = bwz; bwz = temp;
		numTrans /= grouping;

		__syncthreads();
	}

	for (int o = tid; o < (1 << ns[0]); o += blockDim.x)
	{
		bz[o] = (brz[o] % p);
	}
}

void generatePolynomialOnDevice(int n, int p, int **d_poly)
{
	static curandState *states = NULL;
	int *d_out;

	cudaMalloc((void**)&d_out, sizeof(int)*n*2);
	cudaMalloc((void**)&states, sizeof(curandState)*n);

	dim3 block = 64;
	dim3 grid = (n + block.x - 1) / block.x;

	rand_device_api_kernel<<<grid, block>>>(states, d_out, n, p);
	cudaDeviceSynchronize();

	*d_poly = d_out;
}

void powsOf2(int wn, int *pows, int lgn, int n, int p)
{
	for(int i = 1; i <= lgn; i++)
	{
		for(int j = 0; j < n; j++)
		{
			pows[i * n + j] = fastexp(fastexp(wn, (1 << (i-1)), p), j , p);
		}

	}

//	for (int i = 1; i <= lgn; i++)
//	{
//		for(int j = 0; j < n; j++)
//		{
//			printf("[%d, %d %d] ", pows[i * n + j], i, j);
//		}
//		printf("\n");
//	}


}

void writeToFile(int *a, int n)
{
	std::ofstream out;
	out.open("/home/othmane/Dokumente/poly.dat");

	for(int i = 0; i < n; i++)
		out << a[i] << std::endl;

	out.close();
}

void moveToHost(int *d_Buf, int **Buf, int n)
{
	*Buf = (int*)malloc(n*sizeof(int));
	cudaMemcpy(*Buf, d_Buf, n*sizeof(int), cudaMemcpyDeviceToHost);
}

double cpuSecond()
{
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

void multiplyOnCPU(int *A, int *B, int n, int* rC, int p)
{
	int *C = (int*)calloc(2 * n, sizeof(int));

	for (int i = 0; i<n; ++i)
	for (int j = 0; j < n; ++j)
	{
		C[j + i] = (C[j + i] + ((long)B[i] * A[j])) % p;
	}

	for (int i = 0; i < n; ++i)
		C[i] = ((((long)C[i] - C[i + n]) % p) + p) % p;

	for (int i = 0; i < n; ++i)
		rC[i] = C[i];

	free(C);
}

void montgomeryMultiply(int *d_a, int *d_b, int *c, int n, int p, int grid)
{
	long *d_c, *h_c;

	h_c = (long*)malloc(n * sizeof(long));
	cudaMalloc((void**)&d_c, n * grid * sizeof(long));

	// calculate threshold
	long square   = (long)p * p;
	long thres    = (((1UL << 63) - 1) / square);

	dim3 block = (n > 1024) ? 1024 : n;


	double iStart = cpuSecond();
	montgomeryKernel<<<grid, block>>>(d_a, d_b, d_c, n, p, (int)thres);
	cudaDeviceSynchronize();
	double iElaps = cpuSecond() - iStart;

	std::cout << "Time elapsed Montgomery: " << iElaps << std::endl;


	cudaMemcpy(h_c, d_c, n * sizeof(long), cudaMemcpyDeviceToHost);

	for(int i = 0; i < n; i++)
		c[i] = (int)h_c[i];

	free(h_c);
	cudaFree(d_c);
}

void nttMultiply(int *d_a, int *d_b, int *c, int n, int ni, int wn, int wni, int p, int grid)
{
	int *d_c, *d_wn, *d_wni, *wns, *wnis;
	int lgn = (int)round(log2((double)n));

	wns  = (int*)malloc((lgn+1) * (n/2) * sizeof(int));
	wnis = (int*)malloc((lgn+1) * (n/2) * sizeof(int));
	powsOf2(wn, wns, lgn, n/2, p);
	powsOf2(wni, wnis, lgn, n/2, p);

	cudaMalloc((void**)&d_wn,  (lgn+1) * (n/2) * sizeof(int));
	cudaMalloc((void**)&d_wni, (lgn+1) * (n/2) * sizeof(int));

	cudaMalloc((void**)&d_c, n * 2 * sizeof(int));

	cudaMemcpy(d_wn, wns, (lgn+1)  * (n/2) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_wni, wnis, (lgn+1) * (n/2) * sizeof(int), cudaMemcpyHostToDevice);

	dim3 block = ((n / 2) > 1024) ? 1024 : n / 2;


	double iStart = cpuSecond();
	nttKernel<<<1, block>>>(d_a, d_b, d_c, n, d_wn, d_wni, p, ni);
	cudaDeviceSynchronize();
	double iElaps = cpuSecond() - iStart;

	std::cout << "Time elapsed NTT: " << iElaps << std::endl;


	cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);

//	for(int i=0; i<256; i++)
//		c[i] = (((c[i] * 256) % 257) + 257) % 257;
//
//	for(int i=0; i<128; i++)
//		c[i] = (((c[i] - c[i + 128]) % 257) + 257) % 257;


	cudaFree(d_c);
	cudaFree(d_wn);
	cudaFree(d_wni);

	free(wns);
	free(wnis);
}

void nussbaumerMultiply(int *d_a, int *d_b, int *c, int n, int p, int grid, long inv2)
{

	n                = (int)round(log2((float)n));
	int lgn          = (int)ceil(log2((float)n));
	int *ns          = (int*)malloc(lgn * sizeof(int));
	int size         = 1 << n;
	int scalableSize = (n & 0x1) ? (1 << ((n >> 1) + 1)) : (1 << (n >> 1));
	int scratchSize  = size * (1 << lgn);

	ns[0] = n;
	for (int i = 1; i < lgn; i++)
		ns[i] = (ns[i - 1] & 0x1) ? (ns[i-1] >> 1) + 1 : (ns[i-1] >> 1);

	int *d_rx, *d_ry, *d_wx, *d_wy, *d_ns, *d_z, *z;

	cudaMalloc((void**)&d_rx, scratchSize * grid * sizeof(int));
	cudaMalloc((void**)&d_ry, scratchSize * grid * sizeof(int));
	cudaMalloc((void**)&d_wx, scratchSize * grid * sizeof(int));
	cudaMalloc((void**)&d_wy, scratchSize * grid * sizeof(int));
	cudaMalloc((void**)&d_ns, lgn * sizeof(int));
	cudaMalloc((void**)&d_z, size * grid * sizeof(int));

	z = (int*)malloc(size * sizeof(int));

	cudaMemcpy(d_rx, d_a, size * sizeof(int), cudaMemcpyDeviceToDevice);
	cudaMemcpy(d_ry, d_b, size * sizeof(int), cudaMemcpyDeviceToDevice);
	cudaMemcpy(d_ns, ns,  lgn  * sizeof(int), cudaMemcpyHostToDevice);


	double iStart = cpuSecond();
	nussbaumerKernel<<<1, scalableSize>>>(d_rx, d_ry, d_ns, lgn, d_wx, d_wy, scratchSize, d_z, p, inv2);
	cudaDeviceSynchronize();
	double iElaps = cpuSecond() - iStart;

	std::cout << "Time elapsed Nussbaumer: " << iElaps << std::endl;


	cudaMemcpy(z, d_z , size * sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 0; i < size; i++)
		c[i] = z[i];

	free(z);
	free(ns);

	cudaFree(d_rx);
	cudaFree(d_ry);
	cudaFree(d_wx);
	cudaFree(d_wy);
	cudaFree(d_z);

}

void initialize(int **d_a, int **d_b, int **c, int **c_ref, int n, int p)
{
	generatePolynomialOnDevice(n, p, d_a);
	generatePolynomialOnDevice(n, p, d_b);

	*c     = (int*)malloc(n * 2 * sizeof(int));
	*c_ref = (int*)malloc(n * sizeof(int));
}

int main()
{
//	int n = 128;
//	int p = 257;
//
//	int ni  = 256;
//	int wn  = 3;
//	int wni = 86;

//	int n = 16;
//	int p = 786433;
//
//	int ni  = 761857;
//	int wn  = 41596;
//	int wni = 116823;

//	int n = 256;
//	int p = 7681;
//
//	int ni  = 7666;
//	int wn  = 62;
//	int wni = 1115;
//
//	long inv2 = 3841;

	/* params for n = 1024 */
	int n     = 1024;
	int p     = 786433;
	int ni    = 786049;
	int wn    = 19;
	int wni   = 579477;
	long inv2 = 393217;

	/* params for n = 2048*/
//	int n     = 2048;
//	int p     = 786433;
//	int ni    = 786241;
//	int wn    = 14;
//	int wni   = 280869;
//	long inv2 = 393217;

	/* params for n = 4096 */
//	int n     = 4096;
//	int p     = 786433;
//	int ni    = 786337;
//	int wn    = 804;
//	int wni   = 292467;
//	long inv2 = 393217;

	int grid = 1;

	int *d_a, *d_b, *a, *b, *c, *c_ref;
	initialize(&d_a, &d_b, &c, &c_ref, n, p);

	moveToHost(d_a, &a, n);
	moveToHost(d_b, &b, n);

	montgomeryMultiply(d_a, d_b, c, n, p, grid);

	nussbaumerMultiply(d_a, d_b, c, n, p, grid, inv2);

	nttMultiply(d_a, d_b, c, 2*n, ni, wn, wni, p, grid);

	multiplyOnCPU(a, b, n, c_ref, p);

	for(int i = 0; i < n; i++)
		if(c[i] != c_ref[i])
			printf("Error at %d, %d %d\n", i, c_ref[i], c[i]);


	free(a);
	free(b);
	free(c);
	free(c_ref);
	cudaFree(d_a);
	cudaFree(d_b);

	return 0;
}
