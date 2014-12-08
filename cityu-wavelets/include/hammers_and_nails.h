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

	SmartArray():len(0) {}
	// TODO: Need to take care of copy (smart ptr)
	SmartArray(int d):len(d)
	{
		if (len > 0)
		{
			T *blk = new T[len]();
			p = tr1::shared_ptr<T>(blk, array_deleter<T>());
		}
	}

	SmartArray(int d, const T *data):len(d)
	{
		if (len > 0 && data != NULL)
		{
			T *copy = new T[len]();
			//memcpy((void*)copy, (void*)data, sizeof(T) * dims); //TODO: Wrong Doing! Should copy by copy-constructor.
//			T *cast_data = (T*)data;
//			for (int i = 0; i < len; ++i)
//			{
//				copy[i] = cast_data[i];
//			}
			std::copy(data, data + len, copy);
			p = tr1::shared_ptr<T>(copy, array_deleter<T>());
		}
	}

	~SmartArray() {}

	inline void reserve(int n)
	{
		if (n > 0)
		{
			len = n;
			T *blk = new T[len]();
			p = tr1::shared_ptr<T>(blk, array_deleter<T>());
		}
	}

	inline int size() { return len; }

    inline const T& operator[](int i) const { return p.get()[i]; }
    inline T& operator[](int i) { return p.get()[i]; }
    inline operator const T*() const { return p.get(); }
    inline bool operator == (const SmartArray& sz) const
	{
		T *pp0 = p.get(), *pp1 = sz.p.get();
		int d = len, dsz = sz.len;
		if( d != dsz )
			return false;
		if( d == 2 )
			return pp0[0] == pp1[0] && pp0[1] == pp1[1];

		for( int i = 0; i < d; i++ )
			if( pp0[i] != pp1[i] )
				return false;
		return true;
	}

    inline bool operator != (const SmartArray& sz) const {return !(*this == sz);}

    SmartArray clone() const {
    	SmartArray deepcpy(this->len, this->p.get());
    	return deepcpy;
    }

    int len;
    tr1::shared_ptr<T> p;
};

typedef SmartArray<int> SmartIntArray;
typedef SmartArray<double> Smart64FArray;

#endif
