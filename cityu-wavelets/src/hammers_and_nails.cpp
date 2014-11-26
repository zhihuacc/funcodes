#include "../include/hammers_and_nails.h"

//inline SmartArray::SmartArray(int d) : dims(d)
//{
//	if (dims > 0)
//	{
//		p = tr1::shared_ptr<int>(new int[dims], array_deleter<int>());
//	}
//}
//
//inline SmartArray::SmartArray(int d, int *data):dims(d)
//{
//	if (dims > 0 && data != NULL)
//	{
//		int *copy = new int[dims];
//		memcpy((void*)copy, (void*)data, sizeof(int) * dims);
//	}
//}
//
//inline SmartArray::~SmartArray()
//{
//}
//
//inline const int& SmartArray::operator[](int i) const { return p.get()[i]; }
//inline int& SmartArray::operator[](int i) { return p.get()[i]; }
//inline SmartArray::operator const int*() const { return p.get(); }
//
//inline bool SmartArray::operator == (const SmartArray& sz) const
//{
//	int *pp0 = p.get(), *pp1 = sz.p.get();
//    int d = dims, dsz = sz.dims;
//    if( d != dsz )
//        return false;
//    if( d == 2 )
//        return pp0[0] == pp1[0] && pp0[1] == pp1[1];
//
//    for( int i = 0; i < d; i++ )
//        if( pp0[i] != pp1[i] )
//            return false;
//    return true;
//}
//
//inline bool SmartArray::operator != (const SmartArray& sz) const
//{
//    return !(*this == sz);
//}
