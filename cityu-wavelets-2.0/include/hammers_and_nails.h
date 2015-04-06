#ifndef _HAMMERS_AND_NAILS_H
#define _HAMMERS_AND_NAILS_H

//#include <tr1/memory>
#include <memory>
#include <cstring>
#include <map>
#include <string>

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
	SmartArray(int d):len(d)
	{
		if (len > 0)
		{
			// Make sure constructors are invoked.
			T *blk = new T[len]();
			p = shared_ptr<T>(blk, array_deleter<T>());
		}
	}

	SmartArray(int d, const T &k):len(d)
	{
		if (len > 0)
		{
			T *blk = new T[len]();
			p = shared_ptr<T>(blk, array_deleter<T>());
			for (int i = 0; i < len; ++i)
			{
				p.get()[i] = k;
			}
		}
	}

	SmartArray(int d, const T *data):len(d)
	{
		if (len > 0 && data != NULL)
		{
			// Make sure that copy constructors are invoked.
			T *copy = new T[len]();
			std::copy(data, data + len, copy);
			p = shared_ptr<T>(copy, array_deleter<T>());
		}
	}

	~SmartArray() {}

	inline void reserve(int n)
	{
		if (n != len)
		{
			len = n;
			T *blk = new T[len]();
			p = shared_ptr<T>(blk, array_deleter<T>());
		}
	}

	inline int size() const { return len; }

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

    inline SmartArray clone() const {
    	SmartArray deepcpy(this->len, this->p.get());
    	return deepcpy;
    }

    inline void copy(SmartArray &dst)
    {
    	if (len != dst.size())
    	{
    		return;
    	}
    	T *tmp = p.get();
    	for (int i = 0; i < len; ++i)
    	{
    		dst[i] = tmp[i];
    	}
    }

    static inline SmartArray konst(int s, const T &k)
    {
    	SmartArray array(s, k);
    	return array;
    }

    int len;
    shared_ptr<T> p;
};

typedef SmartArray<int> SmartIntArray;
typedef SmartArray<double> Smart64FArray;

int string_options_parser(const string &opt_str, map<string, string> opt_map);
string trim_head_and_tail(const string &str);

clock_t tic();
string show_elapse(clock_t t);

#endif
