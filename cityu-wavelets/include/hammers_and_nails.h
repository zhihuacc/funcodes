#ifndef _HAMMERS_AND_NAILS_H
#define _HAMMERS_AND_NAILS_H

#include <tr1/memory>
#include <cstring>

using namespace std;

template<typename T>
struct array_deleter
{
	void operator() (T const *p)
	{
		delete [] p;
	}
};

template<class T>
struct SmartArray
{

	SmartArray():dims(0) {}
	// TODO: Need to take care of copy (smart ptr)
	SmartArray(int d):dims(d)
	{
		if (dims > 0)
		{
			T *blk = new T[dims];
			memset((void*)blk, 0, sizeof(T) * dims);
			p = tr1::shared_ptr<T>(blk, array_deleter<T>());
		}
	}

	SmartArray(int d, const T *data):dims(d)
	{
		if (dims > 0 && data != NULL)
		{
			T *copy = new T[dims];
			memcpy((void*)copy, (void*)data, sizeof(T) * dims);
			p = tr1::shared_ptr<T>(copy, array_deleter<T>());
		}
	}

	~SmartArray() {}

    const T& operator[](int i) const { return p.get()[i]; }
    T& operator[](int i) { return p.get()[i]; }
    operator const T*() const { return p.get(); }
    bool operator == (const SmartArray& sz) const
	{
		T *pp0 = p.get(), *pp1 = sz.p.get();
		int d = dims, dsz = sz.dims;
		if( d != dsz )
			return false;
		if( d == 2 )
			return pp0[0] == pp1[0] && pp0[1] == pp1[1];

		for( int i = 0; i < d; i++ )
			if( pp0[i] != pp1[i] )
				return false;
		return true;
	}

    bool operator != (const SmartArray& sz) const {return !(*this == sz);}

    SmartArray clone() const {
    	SmartArray deepcpy(this->dims, this->p.get());
    	return deepcpy;
    }

    int dims;
    tr1::shared_ptr<T> p;
};

typedef SmartArray<int> SmartIntArray;
typedef SmartArray<double> Smart64FArray;

#endif
