#include "theil_sen.h"
#include "stdlib.h"
#include "stddef.h"

#define MAX_ELEMENTOW 4095

static int porownaj_floaty(const void* a, const void* b) {
    float roznica = (*(const float*)a - *(const float*)b);
    return (roznica > 0) - (roznica < 0);
}

static float mediana(float* tablica, size_t dlugosc) {
    qsort(tablica, dlugosc, sizeof(float), porownaj_floaty);
    if (dlugosc % 2 == 0) {
        return (tablica[dlugosc / 2 - 1] + tablica[dlugosc / 2]) / 2.0f;
    } else {
        return tablica[dlugosc / 2];
    }
}

WynikTheilSena TheilSen_Estymator(const float* x, const float* y, size_t dlugosc) {

    float nachylenia[MAX_ELEMENTOW];
    float wyrazy_wolne[MAX_ELEMENTOW];
    size_t indeks_nachylen = 0;
    size_t indeks_wyrazow = 0;

    for (size_t i = 0; i < dlugosc - 1; i++) {
    for (size_t j = i + 1; j < dlugosc; j++) {
        if (x[j] != x[i]) {
        if (indeks_nachylen < MAX_ELEMENTOW) {
        nachylenia[indeks_nachylen++] = (y[j] - y[i]) / (x[j] - x[i]);
        } 
        else {
            break;
        }
       }
      }
        if (indeks_nachylen >= MAX_ELEMENTOW) break;
    }

    float mediana_nachylenia = mediana(nachylenia, indeks_nachylen);

    for (size_t i = 0; i < dlugosc; i++) {
     if (indeks_wyrazow < MAX_ELEMENTOW) {
       wyrazy_wolne[indeks_wyrazow++] = y[i] - mediana_nachylenia * x[i];
        }
    }
    float mediana_wyrazu = mediana(wyrazy_wolne, indeks_wyrazow);

    WynikTheilSena wynik = {mediana_nachylenia, mediana_wyrazu};
    return wynik;
}
