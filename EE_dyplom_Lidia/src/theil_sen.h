#ifndef THEIL_SEN_H
#define THEIL_SEN_H

#include <stddef.h>

typedef struct {
    float nachylenie;
    float wyraz;
} WynikTheilSena;

WynikTheilSena TheilSen_Estymator(const float* x, const float* y, size_t dlugosc);

#endif 
