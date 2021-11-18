# Sampling

## Simple Sampling

Simple (random) Sampling


```r
library(dplyr)
```

```
## Warning: package 'dplyr' was built under R version 4.0.5
```

```
## 
## Attaching package: 'dplyr'
```

```
## The following objects are masked from 'package:stats':
## 
##     filter, lag
```

```
## The following objects are masked from 'package:base':
## 
##     intersect, setdiff, setequal, union
```

```r
iris_df <- iris
set.seed(1)
sample_n(iris_df, 10)
```

```
##    Sepal.Length Sepal.Width Petal.Length Petal.Width    Species
## 1           5.8         2.7          4.1         1.0 versicolor
## 2           6.4         2.8          5.6         2.1  virginica
## 3           4.4         3.2          1.3         0.2     setosa
## 4           4.3         3.0          1.1         0.1     setosa
## 5           7.0         3.2          4.7         1.4 versicolor
## 6           5.4         3.0          4.5         1.5 versicolor
## 7           5.4         3.4          1.7         0.2     setosa
## 8           7.6         3.0          6.6         2.1  virginica
## 9           6.1         2.8          4.7         1.2 versicolor
## 10          4.6         3.4          1.4         0.3     setosa
```


```r
library(sampling)
```

```
## Warning: package 'sampling' was built under R version 4.0.5
```

```r
# set unique id number for each row 
iris_df$id = 1:nrow(iris_df)

# Simple random sampling with replacement
srswor(10, length(iris_df$id))
```

```
##   [1] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 1
##  [38] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0
##  [75] 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 1 0
## [112] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
## [149] 0 0
```

```r
# Simple random sampling without replacement (sequential method)
srswor1(10, length(iris_df$id))
```

```
##   [1] 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
##  [38] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
##  [75] 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
## [112] 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0
## [149] 0 0
```

```r
# Simple random sampling with replacement
srswr(10, length(iris_df$id))
```

```
##   [1] 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 1 1 0 0 0 0
##  [38] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
##  [75] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0
## [112] 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0
## [149] 0 0
```


```r
library(survey)
```

```
## Warning: package 'survey' was built under R version 4.0.5
```

```
## Loading required package: grid
```

```
## Loading required package: Matrix
```

```
## Warning: package 'Matrix' was built under R version 4.0.5
```

```
## Loading required package: survival
```

```
## Warning: package 'survival' was built under R version 4.0.5
```

```
## 
## Attaching package: 'survival'
```

```
## The following objects are masked from 'package:sampling':
## 
##     cluster, strata
```

```
## 
## Attaching package: 'survey'
```

```
## The following object is masked from 'package:graphics':
## 
##     dotchart
```

```r
data("api")
srs_design <- svydesign(data = apistrat,
                        weights = ~pw, 
                        fpc = ~fpc, 
                        id = ~1)
```

```
## Warning in as.fpc(fpc, strata, ids, pps = pps): `fpc' varies within strata:
## stratum 1 at stage 1
```


```r
library(sampler)
```

```
## Warning: package 'sampler' was built under R version 4.0.5
```

```r
rsamp(albania,
      n = 260,
      over = 0.1, # desired oversampling proportion
      rep = F)
```

```
##           qarku Q_ID        bashkia BAS_ID    zaz
## 1         Vlore   12          Vlore    128 ZAZ 81
## 2         Berat    1          Berat     11 ZAZ 64
## 3        Tirane   11         Tirane    111 ZAZ 35
## 4        Durres    3          Kruje     33 ZAZ 20
## 5         Lezhe    9         Kurbin     91 ZAZ 15
## 6       Elbasan    4        Elbasan     44 ZAZ 48
## 7        Tirane   11         Tirane    111 ZAZ 29
## 8         Berat    1          Berat     11 ZAZ 64
## 9         Vlore   12          Vlore    128 ZAZ 83
## 10        Korce    7          Maliq     74 ZAZ 68
## 11       Tirane   11         Tirane    111 ZAZ 35
## 12         Fier    5        Divjake     51 ZAZ 56
## 13       Tirane   11     Rrogozhine    115 ZAZ 43
## 14        Lezhe    9          Lezhe     92 ZAZ 13
## 15         Fier    5        Lushnje     54 ZAZ 54
## 16       Durres    3         Durres     31 ZAZ 21
## 17      Elbasan    4         Cerrik     42 ZAZ 46
## 18        Berat    1         Kucove     13 ZAZ 63
## 19       Tirane   11         Tirane    111 ZAZ 37
## 20      Elbasan    4        Elbasan     44 ZAZ 48
## 21       Tirane   11         Tirane    111 ZAZ 36
## 22        Lezhe    9        Mirdite     94 ZAZ 14
## 23       Tirane   11         Tirane    111 ZAZ 29
## 24        Korce    7        Kolonje     72 ZAZ 73
## 25        Vlore   12          Vlore    128 ZAZ 83
## 26  Gjirokaster    6       Libohove     64 ZAZ 79
## 27        Berat    1         Kucove     13 ZAZ 63
## 28        Berat    1          Berat     11 ZAZ 64
## 29         Fier    5        Divjake     51 ZAZ 56
## 30      Elbasan    4        Elbasan     44 ZAZ 47
## 31         Fier    5        Lushnje     54 ZAZ 55
## 32         Fier    5          Fier      52 ZAZ 58
## 33        Berat    1        Polican     15 ZAZ 65
## 34       Tirane   11         Tirane    111 ZAZ 34
## 35      Elbasan    4         Cerrik     42 ZAZ 46
## 36       Durres    3         Durres     31 ZAZ 23
## 37       Durres    3         Shijak     36 ZAZ 25
## 38      Elbasan    4       Prrenjas     46 ZAZ 53
## 39        Kukes    8        Tropoje     88  ZAZ 9
## 40       Tirane   11         Tirane    111 ZAZ 33
## 41       Tirane   11         Tirane    111 ZAZ 37
## 42  Gjirokaster    6       Memaliaj     65 ZAZ 76
## 43       Durres    3         Durres     31 ZAZ 23
## 44       Tirane   11         Tirane    111 ZAZ 37
## 45        Berat    1          Berat     11 ZAZ 64
## 46        Vlore   12         Himare    123 ZAZ 86
## 47        Vlore   12          Finiq    122 ZAZ 89
## 48        Korce    7       Pogradec     75 ZAZ 67
## 49       Tirane   11         Tirane    111 ZAZ 34
## 50        Lezhe    9        Mirdite     94 ZAZ 14
## 51        Korce    7          Korce     73 ZAZ 72
## 52      Elbasan    4          Peqin     45 ZAZ 44
## 53        Vlore   12        Sarande    125 ZAZ 88
## 54         Fier    5          Fier      52 ZAZ 57
## 55        Korce    7       Pogradec     75 ZAZ 67
## 56  Gjirokaster    6    Gjirokaster     62 ZAZ 78
## 57        Berat    1          Berat     11 ZAZ 64
## 58        Diber    2        Bulqize     21 ZAZ 18
## 59       Tirane   11         Tirane    111 ZAZ 41
## 60       Durres    3          Kruje     33 ZAZ 20
## 61       Tirane   11         Tirane    111 ZAZ 34
## 62         Fier    5          Fier      52 ZAZ 57
## 63      Shkoder   10        Shkoder    105  ZAZ 5
## 64        Berat    1          Berat     11 ZAZ 64
## 65  Gjirokaster    6        Kelcyre     63 ZAZ 75
## 66      Shkoder   10        Shkoder    105  ZAZ 4
## 67       Tirane   11         Tirane    111 ZAZ 41
## 68         Fier    5        Lushnje     54 ZAZ 55
## 69         Fier    5        Lushnje     54 ZAZ 55
## 70      Elbasan    4        Elbasan     44 ZAZ 47
## 71       Tirane   11         Kavaje    113 ZAZ 42
## 72      Elbasan    4        Elbasan     44 ZAZ 50
## 73      Shkoder   10 Malesi e Madhe    102  ZAZ 1
## 74        Diber    2            Mat     25 ZAZ 16
## 75      Elbasan    4         Cerrik     42 ZAZ 46
## 76        Korce    7          Maliq     74 ZAZ 68
## 77        Korce    7       Pogradec     75 ZAZ 67
## 78        Vlore   12        Sarande    125 ZAZ 88
## 79      Elbasan    4        Elbasan     44 ZAZ 47
## 80         Fier    5        Divjake     51 ZAZ 56
## 81       Tirane   11         Tirane    111 ZAZ 36
## 82        Vlore   12       Selenice    126 ZAZ 85
## 83        Korce    7          Korce     73 ZAZ 72
## 84       Tirane   11         Tirane    111 ZAZ 40
## 85      Shkoder   10        Shkoder    105  ZAZ 2
## 86        Korce    7          Korce     73 ZAZ 71
## 87        Lezhe    9        Mirdite     94 ZAZ 14
## 88        Berat    1         Kucove     13 ZAZ 63
## 89       Tirane   11         Tirane    111 ZAZ 41
## 90        Korce    7          Korce     73 ZAZ 72
## 91        Vlore   12          Finiq    122 ZAZ 89
## 92       Durres    3         Durres     31 ZAZ 24
## 93        Berat    1          Berat     11 ZAZ 64
## 94        Korce    7          Korce     73 ZAZ 71
## 95        Lezhe    9          Lezhe     92 ZAZ 12
## 96       Tirane   11         Tirane    111 ZAZ 34
## 97  Gjirokaster    6         Permet     66 ZAZ 74
## 98      Shkoder   10    Vau i Dejes    108  ZAZ 6
## 99        Korce    7          Korce     73 ZAZ 72
## 100     Shkoder   10        Shkoder    105  ZAZ 2
## 101     Shkoder   10    Vau i Dejes    108  ZAZ 6
## 102       Lezhe    9          Lezhe     92 ZAZ 12
## 103      Durres    3         Durres     31 ZAZ 23
## 104       Berat    1        Skrapar     17 ZAZ 66
## 105     Elbasan    4          Peqin     45 ZAZ 44
## 106      Tirane   11         Tirane    111 ZAZ 29
## 107      Tirane   11         Tirane    111 ZAZ 29
## 108        Fier    5    Mallakaster     55 ZAZ 61
## 109       Diber    2          Diber     22 ZAZ 19
## 110        Fier    5          Fier      52 ZAZ 58
## 111        Fier    5        Divjake     51 ZAZ 56
## 112       Berat    1         Kucove     13 ZAZ 63
## 113      Tirane   11         Tirane    111 ZAZ 36
## 114      Tirane   11         Tirane    111 ZAZ 41
## 115      Tirane   11         Tirane    111 ZAZ 31
## 116       Kukes    8        Tropoje     88  ZAZ 9
## 117      Tirane   11         Tirane    111 ZAZ 35
## 118       Kukes    8          Kukes     83 ZAZ 11
## 119       Berat    1          Berat     11 ZAZ 64
## 120      Durres    3         Durres     31 ZAZ 24
## 121      Tirane   11         Kavaje    113 ZAZ 42
## 122       Lezhe    9         Kurbin     91 ZAZ 15
## 123       Vlore   12         Himare    123 ZAZ 86
## 124     Shkoder   10        Shkoder    105  ZAZ 3
## 125       Diber    2            Mat     25 ZAZ 16
## 126        Fier    5        Lushnje     54 ZAZ 54
## 127        Fier    5          Fier      52 ZAZ 58
## 128       Kukes    8          Kukes     83 ZAZ 11
## 129      Tirane   11         Tirane    111 ZAZ 33
## 130      Tirane   11         Tirane    111 ZAZ 31
## 131     Elbasan    4        Elbasan     44 ZAZ 48
## 132     Elbasan    4         Gramsh     43 ZAZ 51
## 133       Vlore   12          Finiq    122 ZAZ 89
## 134      Tirane   11         Tirane    111 ZAZ 32
## 135       Vlore   12       Selenice    126 ZAZ 85
## 136 Gjirokaster    6         Permet     66 ZAZ 74
## 137     Shkoder   10        Shkoder    105  ZAZ 5
## 138       Berat    1         Kucove     13 ZAZ 63
## 139 Gjirokaster    6         Permet     66 ZAZ 74
## 140       Lezhe    9        Mirdite     94 ZAZ 14
## 141      Tirane   11         Tirane    111 ZAZ 35
## 142      Durres    3         Durres     31 ZAZ 22
## 143       Kukes    8            Has     81 ZAZ 10
## 144     Elbasan    4       Librazhd     48 ZAZ 52
## 145       Vlore   12          Finiq    122 ZAZ 89
## 146       Vlore   12         Himare    123 ZAZ 86
## 147      Tirane   11         Tirane    111 ZAZ 33
## 148       Kukes    8          Kukes     83 ZAZ 11
## 149      Tirane   11         Tirane    111 ZAZ 35
## 150       Korce    7        Kolonje     72 ZAZ 73
## 151      Durres    3         Durres     31 ZAZ 21
## 152      Tirane   11         Tirane    111 ZAZ 34
## 153      Tirane   11         Tirane    111 ZAZ 36
## 154     Elbasan    4        Elbasan     44 ZAZ 48
## 155     Shkoder   10    Vau i Dejes    108  ZAZ 6
## 156       Vlore   12          Vlore    128 ZAZ 82
## 157      Tirane   11          Kamez    112 ZAZ 27
## 158     Elbasan    4          Peqin     45 ZAZ 44
## 159      Tirane   11         Tirane    111 ZAZ 36
## 160 Gjirokaster    6    Gjirokaster     62 ZAZ 78
## 161      Durres    3         Durres     31 ZAZ 24
## 162      Tirane   11         Tirane    111 ZAZ 38
## 163      Tirane   11         Kavaje    113 ZAZ 42
## 164       Vlore   12          Vlore    128 ZAZ 84
## 165      Tirane   11         Tirane    111 ZAZ 32
## 166        Fier    5        Divjake     51 ZAZ 56
## 167      Tirane   11         Tirane    111 ZAZ 32
## 168      Tirane   11         Tirane    111 ZAZ 29
## 169      Durres    3         Durres     31 ZAZ 24
## 170     Shkoder   10        Shkoder    105  ZAZ 5
## 171       Vlore   12          Vlore    128 ZAZ 81
## 172       Berat    1        Polican     15 ZAZ 65
## 173        Fier    5        Divjake     51 ZAZ 56
## 174      Durres    3         Durres     31 ZAZ 22
## 175      Durres    3         Durres     31 ZAZ 21
## 176       Vlore   12          Vlore    128 ZAZ 83
## 177       Berat    1          Berat     11 ZAZ 64
## 178       Vlore   12          Finiq    122 ZAZ 89
## 179      Tirane   11          Kamez    112 ZAZ 28
## 180      Tirane   11         Tirane    111 ZAZ 34
## 181      Tirane   11         Tirane    111 ZAZ 31
## 182     Shkoder   10        Shkoder    105  ZAZ 2
## 183        Fier    5          Fier      52 ZAZ 57
## 184      Tirane   11         Tirane    111 ZAZ 30
## 185       Lezhe    9        Mirdite     94 ZAZ 14
## 186 Gjirokaster    6       Tepelene     68 ZAZ 77
## 187     Shkoder   10           Puke    104  ZAZ 7
## 188       Lezhe    9          Lezhe     92 ZAZ 13
## 189       Berat    1         Kucove     13 ZAZ 63
## 190       Berat    1        Polican     15 ZAZ 65
## 191       Vlore   12          Vlore    128 ZAZ 81
## 192       Vlore   12          Vlore    128 ZAZ 84
## 193       Korce    7          Korce     73 ZAZ 72
## 194       Korce    7          Korce     73 ZAZ 72
## 195       Berat    1        Skrapar     17 ZAZ 66
## 196     Elbasan    4          Peqin     45 ZAZ 44
## 197      Tirane   11         Tirane    111 ZAZ 33
## 198       Vlore   12        Sarande    125 ZAZ 88
## 199        Fier    5          Fier      52 ZAZ 57
## 200       Vlore   12          Vlore    128 ZAZ 84
## 201     Elbasan    4          Peqin     45 ZAZ 44
## 202       Vlore   12       Selenice    126 ZAZ 85
## 203       Berat    1          Berat     11 ZAZ 64
## 204       Korce    7         Devoll     71 ZAZ 70
## 205        Fier    5        Lushnje     54 ZAZ 54
## 206      Durres    3          Kruje     33 ZAZ 20
## 207       Korce    7          Maliq     74 ZAZ 68
## 208       Korce    7       Pogradec     75 ZAZ 67
## 209     Shkoder   10        Shkoder    105  ZAZ 2
## 210     Elbasan    4         Gramsh     43 ZAZ 51
## 211        Fier    5          Fier      52 ZAZ 57
## 212      Tirane   11          Kamez    112 ZAZ 28
## 213 Gjirokaster    6       Tepelene     68 ZAZ 77
## 214     Elbasan    4        Elbasan     44 ZAZ 47
## 215      Durres    3          Kruje     33 ZAZ 20
## 216      Durres    3         Durres     31 ZAZ 22
## 217       Vlore   12          Vlore    128 ZAZ 81
## 218      Tirane   11         Tirane    111 ZAZ 30
## 219     Shkoder   10        Shkoder    105  ZAZ 2
## 220     Elbasan    4        Elbasan     44 ZAZ 47
## 221     Elbasan    4         Gramsh     43 ZAZ 51
## 222      Tirane   11         Tirane    111 ZAZ 30
## 223      Durres    3         Shijak     36 ZAZ 25
## 224      Tirane   11         Tirane    111 ZAZ 32
## 225      Durres    3         Durres     31 ZAZ 21
## 226       Diber    2          Diber     22 ZAZ 19
## 227     Shkoder   10  Fushe - Arrez    101  ZAZ 8
## 228       Lezhe    9        Mirdite     94 ZAZ 14
## 229        Fier    5        Lushnje     54 ZAZ 54
## 230       Lezhe    9         Kurbin     91 ZAZ 15
## 231     Elbasan    4        Elbasan     44 ZAZ 50
## 232       Kukes    8            Has     81 ZAZ 10
## 233       Korce    7       Pogradec     75 ZAZ 67
## 234     Elbasan    4       Librazhd     48 ZAZ 52
## 235        Fier    5    Mallakaster     55 ZAZ 61
## 236      Tirane   11         Tirane    111 ZAZ 30
## 237     Elbasan    4          Peqin     45 ZAZ 44
## 238        Fier    5          Fier      52 ZAZ 58
## 239      Tirane   11          Kamez    112 ZAZ 27
## 240     Shkoder   10 Malesi e Madhe    102  ZAZ 1
## 241        Fier    5        Divjake     51 ZAZ 56
## 242      Durres    3         Durres     31 ZAZ 23
## 243 Gjirokaster    6    Gjirokaster     62 ZAZ 78
## 244     Elbasan    4         Gramsh     43 ZAZ 51
## 245      Tirane   11         Tirane    111 ZAZ 34
## 246       Korce    7          Maliq     74 ZAZ 68
## 247      Tirane   11         Tirane    111 ZAZ 36
## 248       Berat    1         Kucove     13 ZAZ 63
## 249        Fier    5          Patos     56 ZAZ 59
## 250      Tirane   11         Tirane    111 ZAZ 31
## 251      Durres    3         Durres     31 ZAZ 24
## 252     Elbasan    4          Belsh     41 ZAZ 45
## 253      Durres    3          Kruje     33 ZAZ 20
## 254     Shkoder   10        Shkoder    105  ZAZ 3
## 255      Tirane   11         Tirane    111 ZAZ 34
## 256       Lezhe    9        Mirdite     94 ZAZ 14
## 257        Fier    5          Fier      52 ZAZ 57
## 258        Fier    5        Lushnje     54 ZAZ 55
## 259       Vlore   12          Vlore    128 ZAZ 81
## 260     Shkoder   10        Shkoder    105  ZAZ 2
## 261        Fier    5        Lushnje     54 ZAZ 54
## 262     Elbasan    4          Belsh     41 ZAZ 45
## 263      Tirane   11         Tirane    111 ZAZ 36
## 264     Elbasan    4          Peqin     45 ZAZ 44
## 265     Elbasan    4          Peqin     45 ZAZ 44
## 266      Durres    3         Durres     31 ZAZ 24
## 267       Kukes    8        Tropoje     88  ZAZ 9
## 268        Fier    5        Lushnje     54 ZAZ 54
## 269      Tirane   11         Tirane    111 ZAZ 31
## 270        Fier    5        Lushnje     54 ZAZ 54
## 271     Shkoder   10        Shkoder    105  ZAZ 3
## 272       Korce    7         Devoll     71 ZAZ 70
## 273       Korce    7          Maliq     74 ZAZ 68
## 274       Berat    1          Berat     11 ZAZ 64
## 275 Gjirokaster    6        Dropull     61 ZAZ 80
## 276      Tirane   11         Tirane    111 ZAZ 30
## 277      Tirane   11         Tirane    111 ZAZ 36
## 278        Fier    5        Lushnje     54 ZAZ 54
## 279      Tirane   11         Tirane    111 ZAZ 29
## 280       Korce    7          Korce     73 ZAZ 72
## 281     Shkoder   10 Malesi e Madhe    102  ZAZ 1
## 282       Korce    7       Pogradec     75 ZAZ 67
## 283       Diber    2          Diber     22 ZAZ 19
## 284       Lezhe    9          Lezhe     92 ZAZ 13
## 285       Vlore   12         Himare    123 ZAZ 86
## 286      Tirane   11         Tirane    111 ZAZ 32
##                 njesiaAdministrative COM_ID   qvKod zgjedhes meshkuj femra
## 1                           Shushice  12310  "4543"      245     117   128
## 2                             Otllak   1105  "3441"      892     446   446
## 3    Tirane - Njesia Bashkiake Nr. 5  11215 "18455"      914     436   478
## 4                        Fushe Kruje   3203  "1299"      851     449   402
## 5                              Milot   9104  "0869"      408     218   190
## 6                            Elbasan   4104 "23191"      807     395   412
## 7                             Kashar  11207 "15391"      824     417   407
## 8                             Otllak   1105  "3430"      889     460   429
## 9                              Vlore  12312  "4486"      672     333   339
## 10                          Vreshtas   7316  "3788"      656     334   322
## 11   Tirane - Njesia Bashkiake Nr. 5  11215  "1832"      551     254   297
## 12                            Terbuf   5216  "2850"      605     305   300
## 13                        Rrogozhine  11108  "2227"      867     436   431
## 14                          Blinisht   9202  "0692"      471     222   249
## 15                         Karbunare   5211  "2835"      961     521   440
## 16                          Rashbull   3107 "13691"      754     388   366
## 17                            Cerrik   4103  "2510"      748     366   382
## 18                            Kucove   1201 "35271"      670     335   335
## 19   Tirane - Njesia Bashkiake Nr. 7  11215  "1909"      870     405   465
## 20                           Elbasan   4104  "2308"      694     343   351
## 21   Tirane - Njesia Bashkiake Nr. 6  11215  "1864"      672     322   350
## 22                           Rreshen   9305  "0827"      715     355   360
## 23                           Petrele  11211  "1639"      360     186   174
## 24                            Erseke   7203 "40481"      471     244   227
## 25                             Vlore  12312  "4507"      912     439   473
## 26                          Libohove   6107  "4304"      843     420   423
## 27                             Lumas   1104  "3416"      678     354   324
## 28                            Otllak   1105  "3433"       57      57     0
## 29                           Divjake   5204 "29421"      536     291   245
## 30                        Bradashesh   4102  "2396"      369     194   175
## 31                           Lushnje   5214  "2874"      696     345   351
## 32                              Fier   5103  "3002"      981     502   479
## 33                           Polican   1306 "35791"      507     265   242
## 34   Tirane - Njesia Bashkiake Nr. 4  11215 "18133"      735     357   378
## 35                            Cerrik   4103  "2512"      694     339   355
## 36                           Durres    3101  "1461"      653     304   349
## 37                           Maminas   3105  "1341"      542     278   264
## 38                             Qukes   4308  "2776"      914     474   440
## 39                           Tropoje   8308  "0510"      369     191   178
## 40   Tirane - Njesia Bashkiake Nr. 3  11215  "1767"      710     327   383
## 41   Tirane - Njesia Bashkiake Nr. 7  11215 "19182"      903     444   459
## 42                          Luftinje   6306  "3399"      209     111    98
## 43                           Durres    3101  "1476"      768     374   394
## 44   Tirane - Njesia Bashkiake Nr. 7  11215  "1887"      767     394   373
## 45                            Berat    1101  "3281"      612     313   299
## 46                     Hore Vranisht  12313  "4593"      488     250   238
## 47                             Aliko  12201  "4694"      459     238   221
## 48                           Bucimas   7401  "3881"      577     302   275
## 49   Tirane - Njesia Bashkiake Nr. 4  11215  "1809"      962     436   526
## 50                           Kacinar   9302  "0792"      689     368   321
## 51                             Korce   7303 "36651"      927     444   483
## 52                          Perparim   4405  "2283"      587     303   284
## 53                            Ksamil  12204  "4681"      979     522   457
## 54                           Frakull   5104  "3165"      893     466   427
## 55                         Proptisht   7406  "3947"      353     177   176
## 56                       Gjirokaster   6105 "42422"      767     391   376
## 57                            Berat    1101  "3265"      815     401   414
## 58                     Fushe Bulqize   2103  "1041"      625     322   303
## 59  Tirane - Njesia Bashkiake Nr. 11  11215 "20211"      934     472   462
## 60                     Koder Thumane   3204 "13101"      551     292   259
## 61   Tirane - Njesia Bashkiake Nr. 4  11215  "1802"      807     397   410
## 62                     Qender - FIER   5112  "3135"      986     478   508
## 63                           Shkoder  10312  "0281"      753     369   384
## 64                            Berat    1101 "33111"      753     365   388
## 65                              Suke   6209  "4222"      371     195   176
## 66                           Shkoder  10312 "02521"      476     231   245
## 67  Tirane - Njesia Bashkiake Nr. 11  11215  "2017"      743     381   362
## 68                           Lushnje   5214 "28581"      468     234   234
## 69                           Lushnje   5214 "28721"      616     292   324
## 70                     Labinot fushe   4114  "2422"      716     375   341
## 71                            Kavaje  11104 "21961"      631     315   316
## 72                           Elbasan   4104  "2338"      976     482   494
## 73           Qender - MALESI E MADHE  10105  "0050"      247     128   119
## 74                               Lis   2307  "1008"      615     329   286
## 75                            Shales   4119  "2591"      624     326   298
## 76                             Pojan   7311  "3783"      805     414   391
## 77                         Proptisht   7406  "3948"      295     149   146
## 78                           Sarande  12208 "46712"      988     501   487
## 79                            Funare   4106  "2398"      493     263   230
## 80                             Remas   5215  "2982"      340     183   157
## 81   Tirane - Njesia Bashkiake Nr. 6  11215  "1862"      876     425   451
## 82                             Armen  12301  "4520"      897     485   412
## 83                             Korce   7303 "36501"      522     243   279
## 84  Tirane - Njesia Bashkiake Nr. 10  11215  "1990"      876     426   450
## 85                             Shale  10311  "0118"      622     320   302
## 86                           Drenove   7301  "3709"      534     269   265
## 87                               Fan   9301  "0774"      459     234   225
## 88                           Perondi   1203 "35141"      544     276   268
## 89  Tirane - Njesia Bashkiake Nr. 11  11215 "20231"      588     298   290
## 90                             Korce   7303  "3664"      761     371   390
## 91                         Mesopotam  12103  "4643"      251     116   135
## 92                           Durres    3101  "1457"      720     353   367
## 93                         Velabisht   1111  "3326"      542     293   249
## 94                         Voskopoje   7315  "3746"      508     256   252
## 95                             Lezhe   9206  "0738"      859     413   446
## 96   Tirane - Njesia Bashkiake Nr. 4  11215 "17892"      641     314   327
## 97                           Carcove   6202  "4142"      345     175   170
## 98                            Bushat  10304  "0156"      702     351   351
## 99                             Korce   7303  "3641"      918     453   465
## 100                        Rrethinat  10310  "0113"      647     330   317
## 101                        Vau Dejes  10316  "0178"      985     526   459
## 102                          Balldre   9201  "0682"      662     345   317
## 103                          Durres    3101 "14722"      767     372   395
## 104                          Gjerbes   1304  "3566"      264     140   124
## 105                         Perparim   4405  "2285"      270     142   128
## 106                         Baldushk  11201  "2068"      268     146   122
## 107                           Kashar  11207 "15301"      522     271   251
## 108                           Ballsh   5302 "32231"      462     236   226
## 109                      Zall Dardhe   2214  "1235"      584     303   281
## 110                             Fier   5103  "3017"      775     385   390
## 111                            Remas   5215  "2981"      547     289   258
## 112                          Perondi   1203  "3508"      503     259   244
## 113  Tirane - Njesia Bashkiake Nr. 6  11215 "18681"      746     352   394
## 114 Tirane - Njesia Bashkiake Nr. 11  11215 "20321"      961     497   464
## 115  Tirane - Njesia Bashkiake Nr. 1  11215  "1681"      980     458   522
## 116                            Bytyc   8303  "0458"      289     145   144
## 117  Tirane - Njesia Bashkiake Nr. 5  11215  "1855"      884     447   437
## 118                         Terthore   8212  "0611"      382     198   184
## 119                          Roshnik   1107  "3454"      258     138   120
## 120                          Durres    3101  "1425"      716     356   360
## 121                      Luz i vogel  11107  "2219"      838     451   387
## 122                         Mamurras   9103  "0913"      666     336   330
## 123                           Himare  12303 "45841"      538     265   273
## 124                          Shkoder  10312 "02361"      487     247   240
## 125                              Baz   2301  "0923"      764     390   374
## 126                           Krutje   5213  "2927"      901     467   434
## 127                             Fier   5103  "2989"      899     451   448
## 128                            Kukes   8207  "0651"      988     474   514
## 129  Tirane - Njesia Bashkiake Nr. 3  11215 "17511"      770     386   384
## 130  Tirane - Njesia Bashkiake Nr. 1  11215  "1697"      988     474   514
## 131                          Elbasan   4104 "23031"      610     305   305
## 132                            Lenie   4205  "2642"      205     106    99
## 133                            Aliko  12201  "4690"      398     197   201
## 134  Tirane - Njesia Bashkiake Nr. 2  11215  "1718"      740     384   356
## 135                         Vllahine  12311  "4557"      900     487   413
## 136                           Permet   6206 "41761"      982     482   500
## 137                          Shkoder  10312  "0275"      672     326   346
## 138                            Lumas   1104  "3420"      264     139   125
## 139                           Permet   6206  "4177"      988     499   489
## 140                              Fan   9301  "0783"      384     200   184
## 141  Tirane - Njesia Bashkiake Nr. 5  11215  "1820"      689     325   364
## 142                          Durres    3101 "14151"      713     356   357
## 143                            Golaj   8103  "0553"      530     267   263
## 144                Qender - LIBRAZHD   4307  "2742"      914     474   440
## 145                         Livadhja  12205  "4714"      554     291   263
## 146                           Himare  12303  "4591"      700     343   357
## 147  Tirane - Njesia Bashkiake Nr. 3  11215  "1763"      739     354   385
## 148                       Gryke-Caje   8204  "0583"      263     121   142
## 149  Tirane - Njesia Bashkiake Nr. 5  11215  "1847"      812     387   425
## 150                         Leskovik   7204 "40511"      501     247   254
## 151                            Sukth   3109  "1363"      998     498   500
## 152  Tirane - Njesia Bashkiake Nr. 4  11215 "18011"      586     283   303
## 153  Tirane - Njesia Bashkiake Nr. 6  11215  "1876"      758     384   374
## 154                          Elbasan   4104  "2309"      886     430   456
## 155                           Shllak  10313  "0131"      319     163   156
## 156                            Vlore  12312  "4505"     1000     512   488
## 157                            Kamez  11206 "15732"      863     443   420
## 158                            Peqin   4404 "22731"      572     288   284
## 159  Tirane - Njesia Bashkiake Nr. 6  11215 "18741"      807     412   395
## 160                         Lunxheri   6108  "4310"      640     320   320
## 161                          Durres    3101  "1442"      787     397   390
## 162  Tirane - Njesia Bashkiake Nr. 8  11215 "19321"      570     269   301
## 163                            Golem  11101 "21271"      614     318   296
## 164                            Vlore  12312 "44751"      473     235   238
## 165  Tirane - Njesia Bashkiake Nr. 2  11215 "17091"      817     406   411
## 166                          Divjake   5204  "2942"      675     323   352
## 167  Tirane - Njesia Bashkiake Nr. 2  11215  "1709"      814     392   422
## 168                          Petrele  11211  "1650"      375     187   188
## 169                          Durres    3101  "1499"      769     386   383
## 170                          Shkoder  10312 "03051"      576     269   307
## 171                   Qender - VLORE  12307 "44211"      490     253   237
## 172                          Polican   1306  "3579"      559     278   281
## 173                         Gradisht   5209  "2957"      664     343   321
## 174                          Durres    3101  "1468"      985     496   489
## 175                            Sukth   3109 "13571"      557     284   273
## 176                            Vlore  12312  "4439"      812     405   407
## 177                           Berat    1101  "3268"      774     388   386
## 178                        Mesopotam  12103  "4640"      939     482   457
## 179                         Paskuqan  11210 "20462"      929     476   453
## 180  Tirane - Njesia Bashkiake Nr. 4  11215 "18021"      705     343   362
## 181  Tirane - Njesia Bashkiake Nr. 1  11215  "1702"      629     312   317
## 182                        Rrethinat  10310  "0104"      696     355   341
## 183                           Portez   5111  "3123"      801     433   368
## 184                            Farke  11205  "2077"      754     393   361
## 185                          Rreshen   9305  "0828"      347     179   168
## 186                         Tepelene   6310 "43911"      462     224   238
## 187                          Gjegjan  10204  "0373"      202     107    95
## 188                            Kolsh   9205  "0724"      289     150   139
## 189                           Kozare   1202  "3501"      219     114   105
## 190                          Polican   1306  "3577"      900     453   447
## 191                         Shushice  12310  "4542"      840     439   401
## 192                            Vlore  12312  "4474"      780     391   389
## 193                            Korce   7303 "36881"      741     365   376
## 194                            Korce   7303  "3649"      827     393   434
## 195                 Qender - SKRAPAR   1308  "3604"      429     231   198
## 196                            Sheze   4406  "2293"      267     143   124
## 197  Tirane - Njesia Bashkiake Nr. 3  11215 "17581"      650     315   335
## 198                          Sarande  12208  "4675"      903     442   461
## 199                         Dermenas   5102  "3150"      396     211   185
## 200                            Vlore  12312  "4480"      506     262   244
## 201                           Gjocaj   4401  "2247"      917     496   421
## 202                             Kote  12304  "4602"      639     339   300
## 203                           Berat    1101  "3283"      558     283   275
## 204                           Proger   7105  "3755"      598     297   301
## 205                            Dushk   5205  "2817"      631     325   306
## 206                            Kruje   3205  "1269"      869     439   430
## 207                         Vreshtas   7316 "37911"      512     259   253
## 208                          Bucimas   7401  "3884"      992     504   488
## 209                        Rrethinat  10310  "0097"      552     284   268
## 210                           Pishaj   4206  "2658"      347     177   170
## 211                            Levan   5107  "3179"      963     480   483
## 212                         Paskuqan  11210 "20461"      958     512   446
## 213                            Lopes   6305  "4362"      497     263   234
## 214                         Shushice   4121  "2479"      982     490   492
## 215                    Koder Thumane   3204  "1310"      703     372   331
## 216                          Durres    3101 "14592"      644     321   323
## 217                           Orikum  12306 "45151"      442     225   217
## 218                      Zall Bastar  11218  "1667"      591     300   291
## 219                        Rrethinat  10310  "0109"      667     338   329
## 220                       Bradashesh   4102  "2381"      351     182   169
## 221                           Gramsh   4201  "2595"      720     352   368
## 222                             Dajt  11204  "1627"      752     389   363
## 223                        Xhafzotaj   3110 "14041"      633     317   316
## 224  Tirane - Njesia Bashkiake Nr. 2  11215  "1713"      693     337   356
## 225                            Manez   3106  "1350"      393     203   190
## 226                        Maqellare   2207  "1111"      473     240   233
## 227                           Fierze  10202  "0355"      360     176   184
## 228                          Kacinar   9302  "0788"      264     128   136
## 229                        Hysgjokaj   5210  "2830"      677     359   318
## 230                              Lac   9102  "0849"      990     504   486
## 231                          Elbasan   4104  "2355"      994     483   511
## 232                            Golaj   8103  "0545"      227     117   110
## 233                         Trebinje   7407  "3964"      446     243   203
## 234                         Librazhd   4302 "27041"      875     445   430
## 235                           Fratar   5303  "3235"      234     126   108
## 236                        Zall Herr  11219  "2053"      235     128   107
## 237                            Peqin   4404 "22721"      463     232   231
## 238                             Fier   5103  "3003"      641     323   318
## 239                            Kamez  11206  "1590"      965     508   457
## 240          Qender - MALESI E MADHE  10105  "0052"      231     129   102
## 241                           Terbuf   5216  "2855"      731     368   363
## 242                          Durres    3101  "1482"      504     238   266
## 243                         Antigone   6101  "4264"      488     253   235
## 244                            Kukur   4203  "2628"      310     173   137
## 245  Tirane - Njesia Bashkiake Nr. 4  11215 "18031"      915     471   444
## 246                         Vreshtas   7316  "3793"      615     325   290
## 247  Tirane - Njesia Bashkiake Nr. 6  11215  "1870"      911     472   439
## 248                           Kozare   1202  "3500"      458     243   215
## 249                            Patos   5110  "3076"      611     321   290
## 250  Tirane - Njesia Bashkiake Nr. 1  11215  "1693"      682     325   357
## 251                          Durres    3101 "14381"      574     272   302
## 252                            Belsh   4101  "2559"      813     410   403
## 253                             Bubq   3201  "1252"      695     374   321
## 254                          Shkoder  10312 "02991"      473     223   250
## 255  Tirane - Njesia Bashkiake Nr. 4  11215 "17951"      843     416   427
## 256                            Rubik   9306  "0897"      996     501   495
## 257                         Dermenas   5102 "31481"      476     258   218
## 258                          Lushnje   5214  "2856"      859     407   452
## 259                         Shushice  12310  "4539"      684     364   320
## 260                         Postribe  10307 "00771"      480     255   225
## 261                            Dushk   5205  "2812"      634     323   311
## 262                           Fierze   4105  "2564"      927     472   455
## 263  Tirane - Njesia Bashkiake Nr. 6  11215  "1878"      869     437   432
## 264                           Pajove   4403  "2269"      748     390   358
## 265                           Gjocaj   4401  "2244"      735     395   340
## 266                          Durres    3101 "14321"      948     486   462
## 267                            Bujan   8302  "0456"      350     169   181
## 268                            Dushk   5205  "2814"      848     450   398
## 269  Tirane - Njesia Bashkiake Nr. 1  11215  "1683"      794     367   427
## 270                        Karbunare   5211  "2842"      220     118   102
## 271                          Shkoder  10312  "0245"      586     295   291
## 272                           Proger   7105 "37511"      477     235   242
## 273                            Pojan   7311  "3770"      484     255   229
## 274                           Berat    1101  "3273"      956     485   471
## 275                   Dropull posht.   6103  "4278"      920     432   488
## 276                      Zall Bastar  11218  "1670"      297     158   139
## 277  Tirane - Njesia Bashkiake Nr. 6  11215 "18731"      701     333   368
## 278                      Fier Shegan   5206  "2913"      574     296   278
## 279                             Peze  11212  "2099"      942     501   441
## 280                            Korce   7303 "36641"      723     349   374
## 281          Qender - MALESI E MADHE  10105  "0055"      469     237   232
## 282                           Velcan   7408  "3974"      462     250   212
## 283                         Kastriot   2204  "1187"      359     200   159
## 284                            Kolsh   9205  "0725"      899     450   449
## 285                           Himare  12303  "4590"      693     351   342
## 286  Tirane - Njesia Bashkiake Nr. 2  11215  "1720"      808     380   428
##     totalSeats                                                     vendndodhja
## 1           12                                   Beshisht, Qendra Shendetesore
## 2            7                                                 Morave, Shkolla
## 3           34                  Rruga "Robert Zhvarc", Shkolla "At Zef Pllumi"
## 4           14                                        Lokal Privat, Osman Cela
## 5            7                    Mal-Bardhe Fushe, Lokal Privat Kujtim copani
## 6           14                            Lagjia "5 Maj", Shkolla "Mahir Domi"
## 7           34                         Mezez Koder   Shkolla publike 9-vjecare
## 8            7                                      Dushnik, Shkolla 9-vjecare
## 9           12                                    Shkolla "15 Tetori"- Kati II
## 10          11                    Fshati Vreshtas, Shkolla "Jeta e re", kati 1
## 11          34      Rruga "Emin Duraku", Konsultore e femijeve prane stadiumit
## 12          16                                        Cerme e siperme, Shkolla
## 13          34                                          Shkolla 9-vjecare Xh.S
## 14           7                                           Baqel,Shkolla Fillore
## 15          16                          Karbunare e Siperme, Shkolla 8 vjecare
## 16          14                                            Shenavlash,  Shkolle
## 17          14                                      Lagjia 2,Shkolla bujqesore
## 18           7                       Lagja Nr2, Bll "Ll.Prifti", Kopshti Nr. 3
## 19          34                            Shkolla 9-vjecare "Sabaudin Gabrani"
## 20          14                       Lagjia"Haxhias", Kopeshti i femijeve nr 3
## 21          34                                     Shkolla "Lasgush  Poradeci"
## 22           7                               Blloku 5+6, Shkolla "Pashko Vasa"
## 23          34                                                  Hekal, shkolla
## 24          11                                                     Kopshti Nr2
## 25          12                                Shkolla "Naim Frasheri" - Kati I
## 26           5                                              Kopshti Hoxhallare
## 27           7                                        Vodez, Shkolla 9 Vjecare
## 28           7                        Parangoi,Dyshnik,(godina e paraburgimit)
## 29          16                                       Divjake Shkolla 8 vjecare
## 30          14                                             Karakullak, Shkolla
## 31          16                                                    Nd. Komunale
## 32          16                      Mbrostar, shkolla 9-vjecare "L Kostandini"
## 33           7                                                Polican, "Holli"
## 34          34             Rr " Dibres",  Laboratori i Fakultetit te Mjekesise
## 35          14                                          Lagjia 2, Kolegji Turk
## 36          14                                          Shkolla "Jan Kukuzeli"
## 37          14                                                 Karpen, Kopshti
## 38          14                                               Karkavec, Shkolla
## 39           3                                                   Aste, Shkolla
## 40          34 Rruga"Kongresi I Manastirit", Shkolla 9-vjecare "Xhezmi Delli" 
## 41          34                                         Shkolla "Naim Frasheri"
## 42           5                                         Rabie,shkolla 8 vjecare
## 43          14                                            Kopshti "Sotir Noka"
## 44          34                                 Pallati i Sportit "Asllan Rusi"
## 45           7                           Lagjia "J.Vruho", Shkolla TH.Tani" K1
## 46          12                                               Vranisht, Shkolla
## 47          12                                              Qenurja, Ambulance
## 48          11                                       Verdove,Shkolla 9-vjecare
## 49          34                              Rruga e " Dibres ", Kopshti Nr. 39
## 50           7                                        Simon, Shkolla 9-vjecare
## 51          11                                             L 18 Sh. Sotir Gura
## 52          14                                               Perparim, Shkolla
## 53          12                   Ksamil, Lagjja nr.2 Shkolla e mesme bashkuar 
## 54          16                                        Kafaraj, Shkolla e mesme
## 55          11                                       Proptisht Shkolla e mesme
## 56           5                                                Shkolla Zinxhira
## 57           7                                      Lagjia "Mangalen", Shkolla
## 58           6                                           Fushe Bulqize Shkolla
## 59          34                      Rr."Shkodres", Lokal Privat Xhelal Bodini 
## 60          14                                                 Gramez, Shkolla
## 61          34                Rruga " Niko Avrami ",  Shkolla "Ramazan Jarani"
## 62          16                                      clirim , Shkolla 9 vjecare
## 63          11                         Lagjia"M. Alimani", Shkolla e ndertimit
## 64           7                        Lagjia "clirim", Shkolla  " M.A. Vrioni"
## 65           5                                         Gorice, Shkolla fillore
## 66          11                            Lagjia "Naim Gjylbegu", Kolegji turk
## 67          34                   Rr."Vangjel Noti", Shkolla "Skender Luarasi" 
## 68          16                                                       Ambulanca
## 69          16                                     Shkolla Kongresi i Lushnjes
## 70          14                                      Mengel, Shkolla 9 -Vjecare
## 71          34                              Piceri''Taverna'' , Agron Arkaxhiu
## 72          14                     Lagjia Clirim, Lokal privat , "Lefter Hodo"
## 73          11                                                  Flake, Kopesht
## 74           6                                               Burgajet, Shkolla
## 75          14                              Licaj 2, Objekt privat, Nefail Ago
## 76          11                                                 Burimas,Shkolle
## 77          11                                        Verri,  Shkolla. Fillore
## 78          12                                   Lagjja 3, Shkolla " 9 Tetori"
## 79          14                                               Bixillej, Shkolla
## 80          16                                               Adriatik, Shkolla
## 81          34                                         Shkolla "Shyqyri  Peza"
## 82          12                                     Treblove, Shkolla 9-vjecare
## 83          11                                        Gjimnazi "Raqi Qirinxhi"
## 84          34                                                   Sh "K. Pezes"
## 85          11                                                   Lekaj Shkolla
## 86          11                                 Boboshtice, Qender Shendetesore
## 87           7                                    Domgjon, Shkolla 9 - Vjecare
## 88           7                                                  Goraj, Shkolla
## 89          34                Rr."Princ Vidi"Breg Lumi, Nder.Mirmb.Rruga -Ura 
## 90          11                                             L 18 Sh. Sotir Gura
## 91          12                                                Brajlat, Shkolla
## 92          14                                        Shkolla " Leonik Tomeo "
## 93           7                                              Starove 2, Shkolla
## 94          11                                              Voskopoje, Shkolla
## 95           7                      Lagjia "Beslidhja", Shkolla "Kolin Gjoka "
## 96          34                                 Rruga "A.Moisiu", cerdhja Nr 19
## 97           5                          Iljar - Munushtir, Shkolla 8 - vjecare
## 98          11                                     Bushat, Shkolla 9 - Vjecare
## 99          11                                            L.4 Shkolla "N.Naci"
## 100         11                                               Hot i Ri, Shkolla
## 101         11                                             Vau-Dejes,  Shkolla
## 102          7                                        Balldren i Ri, Ambulanca
## 103         14                                  Universiteti Aleksander Moisiu
## 104          7                                       Strafice, Shkolla fillore
## 105         14                                      copanaj,Shkolla 9-vjecare 
## 106         34                                                 Parret, Shkolla
## 107         34                                    Shkolla Publike Katundi I Ri
## 108         16                               Shkolla e  mesme " Dervish Hekal"
## 109          6                                  Zall-Dardhe, Shkolla 9-vjecare
## 110         16                                     Sheq Madh, Shkolla e Sheqit
## 111         16                                       Karavasta e re, Ambulanca
## 112          7                                 Rreth Tapi, shkolla 9 - Vjecare
## 113         34                                       Shkolla "Musine Kokalari"
## 114         34               Koder Kamez, Drejtoria e Trajtimit te Studenteve 
## 115         34                               Kopshti Nr. 56,  Rr."Shemsi Haka"
## 116          3                                                Kepenek, Shkolla
## 117         34                    rruga "Vangjel Koca", Shkolla "Murat Toptani
## 118          3                                                 Morine, Shkolla
## 119          7                                Qafe  Dardhe, Shkolla 9 -Vjecare
## 120         14                                     Lagjia 15, Shkolla "M Hasa"
## 121         34                                    Vorrozen,  Shkolla 9-vjecare
## 122          7                           Zef Hoti, Shkolla 9-vjecare "Migjeni"
## 123         12                               Gjilek, lokali privat "Milo Bica"
## 124         11                                      Shiroke, Shkolla 9 vjecare
## 125          6                                            Baz, Shkolla e mesme
## 126         16                      Fier I Ri, Lokal Privat, Antigoni Gjermeni
## 127         16                        Lagjia "8 Shkurti', Shkolla "P. Ikonomi"
## 128          3                                                cerdhja Lagjia 5
## 129         34  Rruga "Sitki cico", Shkolla 9-vjecare "Niket Dardani"         
## 130         34                                    Shkolla 9-vjecare "Ali Demi"
## 131         14                                 Lagjia "A.Pasha", Kopeshti Nr.4
## 132         14                                          Lenie, Shkolla e mesme
## 133         12                                                Ruhulla, Kulture
## 134         34                    Rr."Mihal Grameno", Shkolla "Mihal Grameno" 
## 135         12                                                 Kocul, Shkolla 
## 136          5                                                           Muzeu
## 137         11                            Lagjia "Perash', Shkolla "Ali Lacej"
## 138          7                                      Luzaj, Shkolla 9 - Vjecare
## 139          5                                 Shkolla 9 vjecare "M.Goshishti"
## 140          7                                    Katund i Ri, Shkolla fillore
## 141         34                    Rruga "Pjeter Bogdani",Shkolla "Edit Durham"
## 142         14                                            Shkolla "Hasan Koci"
## 143          3                                               Vlahen 2, Shkolla
## 144         14                                     Spathare Shkolla 9- Vjecare
## 145         12                                                 Zminec, Shkolla
## 146         12                                                  Himare,Kopshti
## 147         34                Rruga" Kongresi  I Manastirit", Shkolla "cajupi"
## 148          3                                      Shkolla 9 vjecare Buzmadhe
## 149         34                             Rruga "Irfan Tomini", cerdhja Nr.57
## 150         11                                            Shkolla "Jani Vreto"
## 151         14                                               Vadardhe, Shkolla
## 152         34                          Rruga " Niko Avrami ",  Kopshti Nr. 25
## 153         34                                   Shkolla "Kongresi Manastirit"
## 154         14                         Lagjia"Haxhias",Shkolla "Fadil Gurmani"
## 155         11                                                  Bene,  Shkolla
## 156         12                                    Objekt sherbimi Kajtas Likaj
## 157         34            Rruga "Bulevardi Blu" nr.486/1, "Pallati i Kultures"
## 158         14                                          Lagjia "Teqe", Spitali
## 159         34                                                   Kopshti Nr.38
## 160          5                                         Gjat -Erind, ambulanca.
## 161         14                                 Konvikti i Shkolles Fiskultures
## 162         34                                            Shkolla "Misto Mame"
## 163         34                             Golem 2, Shkolla 9- vjecare klasa 2
## 164         12                                       Shkolla"15 Tetori"-Kati I
## 165         34                                         Shkolla "Mihal Grameno"
## 166         16                                       Divjake Shkolla 8 vjecare
## 167         34                                         Shkolla "Mihal Grameno"
## 168         34                                                  Daias, shkolla
## 169         14                                Lokal privat, Kastriot Xhaferraj
## 170         11                            Lagjia "3 Heronjte, Shkolla "N.Mazi"
## 171         12                                   Zvernec,  Shkolla 9 - Vjecare
## 172          7                                       Polican, Shkolla "R Keli"
## 173         16                           Mertish 2, Lokal privat  "Solli Qose"
## 174         14                                   SHKOLLA JO PUBLIKE "RILINDJA"
## 175         14                                              Hidrovore, Shkolla
## 176         12                                 Shkolla  9- vjecare " Rilindja"
## 177          7                                  Lagjia "28 Nentori", Ambulanca
## 178         12                                         Krane, Vatra e Kultures
## 179         34               Rr. "Demokracia",Paskuqan 1, Lokal Privat " Kupa"
## 180         34                Rruga " Niko Avrami ",  Shkolla "Ramazan Jarani"
## 181         34                                    Shkolla "Kushtrimi i Lirise"
## 182         11                                                  Golem, Shkolla
## 183         16                                        Lalar, Shkolla 9 vjecare
## 184         34                                 Shkolla e mesme , Farke e Madhe
## 185          7                                    Bukmire, Shkolla 9 - Vjecare
## 186          5                                                Qendra Kulturore
## 187         11                                      Mesul, Shkolla 9 - Vjecare
## 188          7                                                  Kolsh, Shkolla
## 189          7                                 Fjermimar, Lokali Gezim Xhelili
## 190          7                                       Polican, Shkolla mekanike
## 191         12                                                  Mekat, Shkolla
## 192         12                                            Shkolla "Jani Minga"
## 193         11                                            L13 Shkolla Speciale
## 194         11                                                   Kopshti nr 12
## 195          7                                      Sharove, Shkolla e fshatit
## 196         14                                       Algjinaj, Shkolla Fillore
## 197         34        Rruga "Qemal Stafa", Shkolla 9-vjecare "Hasan Prishtina"
## 198         12                                                Gjashte, Shkolla
## 199         16                                       Lokal privat, Zoi  Konomi
## 200         12                                       Zyra e Ndermarrjes Pyjore
## 201         14                                Gjocaj, Lokal privat, Gezim Topi
## 202         12                                         Kote, Shkolla 9 vjecare
## 203          7                          Lagjia "J.Vruho", Shkolla "TH.Tani" K3
## 204         11                                               Vranisht, Shkolla
## 205         16                         Dushk Peqin Lokal privat, Sherbet Dedej
## 206         14                                        Shkolle 9- vjecare "X.P"
## 207         11                        Fshati Bregas, Shkolla 9-vjecare, kati 1
## 208         11                                    Geshtenjas,Shkolla 9-vjecare
## 209         11                                             Shtoj i Ri, Shkolla
## 210         14                                       Drize, Shtepia e Kultures
## 211         16                                         Levan Shkolla 9 vjecare
## 212         34               Rr. "Demokracia",Paskuqan 1, Lokal Privat " Kupa"
## 213          5                                               Dhemblan, Shkolla
## 214         14                                             Hajdaran, Shkolla  
## 215         14                                                 Gramez, Shkolla
## 216         14                                             Shkolla Fiskultures
## 217         12                    Dukat Fshat, Shkolla 9-vjecare " 28 Nentori"
## 218         34                        Fshati Mener i Siperm, Shkolla 9-vjecare
## 219         11                                                Bardhaj, Shkolla
## 220         14                                                Shtemaj, Shkolla
## 221         14                                             Pallati i  Kultures
## 222         34                                              Lanabregas Shkolla
## 223         14                                               Xhafzotaj,Shkolla
## 224         34                                         Shkolla "Mihal Grameno"
## 225         14                                          Borce, Shkolla fillore
## 226          6                                   Pocest 1, Shkolla 9 - Vjecare
## 227         11                                                  Fierze,Shkolla
## 228          7                                      Kacinar, Shkolla 9-vjecare
## 229         16                                               Hysgjokaj,shkolla
## 230          7                                                   Lac, Spitali 
## 231         14                 L.''Emin Matraxhiu'' shkolla ''Abdyl Paralloi''
## 232          3                                                 Zgjeqe, Shkolla
## 233         11                                               Trebinje, Shkolla
## 234         14                                      Librazhd, Qendra Kulturore
## 235         16                          Malas, lokali privat ,Mirjan Skenderaj
## 236         34                                        Kallmet, Shkolla fillore
## 237         14                               Lagjia "Teqe", Pallati i Kultures
## 238         16                         Mbrostar, Lokal privat Floresha Merkaj.
## 239         34            Rruga "Durresi" nr.99, Objekt Privat, Besim Ibrahimi
## 240         11                               Koplik i Siperm,shkolla 9-vjecare
## 241         16                                Shenepremte, Shkolla 9 - Vjecare
## 242         14                                             Shkolla "Vasil Ziu"
## 243          5                                            Arshi Lengo, Shkolla
## 244         14                                   Grazhdan, Shkolla 9 - Vjecare
## 245         34                    Rr. "Selaudin Bekteshi", Shkolla  9 -Vjecare
## 246         11                      Fshati Podgorie, Shkolla 9-vjecare, kati 1
## 247         34                                  Lokal privat  Ndricim Xhaferaj
## 248          7                                                  Salce, Shkolla
## 249         16                                 Grize-Lenginas, Shkolla Fillore
## 250         34                                      Shkolla 9-vjecare "1 Maji"
## 251         14                              Lagjia 18, Shkolla "Naim Frasheri"
## 252         14                                   Qafshkalle, Shkolla 9-vjecare
## 253         14                                     Mallkuc, Shkolla 9 -vjecare
## 254         11                               Lagjia "Salo Halili", Shkolla ABC
## 255         34                       Rr."Myslim Keta", Shkolla  "Pal Engjelli"
## 256          7                       Rubik,Lagjia e vjeter Shkolla 9 - Vjecare
## 257         16                                        Povelce, Shkolla e Mesme
## 258         16                                             Shkolla 4 Deshmoret
## 259         12                                               Shushice, Shkolla
## 260         11                                         Boks 2 Shkolla  e Mesme
## 261         16                                           Zhame sektor, Shkolla
## 262         14                                 Fierze Fshat, Shkolla 9-vjecare
## 263         34                                            Shkolla "1 Qershori"
## 264         14                                        Grykesh, Shkolla fillore
## 265         14                                      Kurtaj, Shkolla 9-vjecare 
## 266         14                                            Shkolla "Demokracia"
## 267          3                                                 Markaj, Shkolla
## 268         16                                             Thanasaj, Ambulanca
## 269         34                           Sh. "Kushtrimi Lirise" Rr. "Ali Demi"
## 270         16                                             Bicakaj , Ambulanca
## 271         11                                Lagjia Kiras, Shkolla Veterinare
## 272         11                                                Cangonj, Kopshti
## 273         11                                              Pendavinje,Shkolle
## 274          7                          Lagjia "30 Vjetori", Stadiumi "Tomori"
## 275          5                                     Goranxi, Shtepia e Kultures
## 276         34                        Fsh.Bastar Murrizi, Shkollla 9 - vjecare
## 277         34                                          Shkolla "Myslym  Keta"
## 278         16                                                  Thane, Shkolla
## 279         34                                            Peze e madhe,Shkolla
## 280         11                                             L 18 Sh. Sotir Gura
## 281         11                                         Jubice, Shkolla fillore
## 282         11                                 Bishnic Nr.6. Shkolla 9-vjecare
## 283          6                                            Borovjan, Konsultore
## 284          7                                                  Gryke, Shkolla
## 285         12                                                  Himare,Shkolla
## 286         34                     rruga "Ali Visha", Shkolla "Osman Myderizi"
##     ambienti totalVoters femVoters maleVoters unusedBallots damagedBallots
## 1     Publik          95        49         46           154              0
## 2     Publik         369       183        186           538              2
## 3     Publik         516       240        276           413              3
## 4     Privat         446       190        256           422              0
## 5     Privat         223        99        124           190              3
## 6     Publik         387       188        199           435              1
## 7     Publik         436       200        236           400              4
## 8     Publik         487       228        259           417              2
## 9     Publik         287       152        135           396              2
## 10    Publik         319       157        162           348              2
## 11    Publik         310       160        150           251              1
## 12    Publik         290       140        150           327              0
## 13    Publik         350         0        350           534              0
## 14    Publik         236       125        111           244              0
## 15    Publik         529       215        314           451              0
## 16    Publik         393       180        213           376              0
## 17    Publik         305       144        161           305              0
## 18    Publik         246       114        132           437              0
## 19    Publik         470       244        226           416              1
## 20    Publik         329       157        172           379              0
## 21    Publik         321       161        160           362              2
## 22    Publik         364       193        171             0              0
## 23    Publik         265       122        143            97              5
## 24    Publik         255       115        140           224              1
## 25    Publik         286       145        141           643              1
## 26    Publik         443       216        227           414              2
## 27    Publik         362       169        193           329              0
## 28   Posacem          61         0         61            62              0
## 29    Publik         248        98        150           297              1
## 30    Publik         275       123        152           100              1
## 31    Publik         302       150        152           420              1
## 32    Publik         440         0        440           560              0
## 33    Publik         280       137        143           236              1
## 34    Publik         446       224        222           302              1
## 35    Publik         237       106        131           469              1
## 36    Publik         268       138        130           432              0
## 37    Publik         266        97        169           286              0
## 38    Publik         473       230        243           459              0
## 39    Publik         198        95        103           177              1
## 40    Publik         348       175        173           376              0
## 41    Publik         468       221        247           451              2
## 42    Publik          40         8         32           173              0
## 43    Publik         335       161        174           448              0
## 44    Publik         375       179        196           407              0
## 45    Publik         301       141        160             0              0
## 46    Publik         223       101        122           274              0
## 47    Publik         112         0        112           357              1
## 48    Publik         265       116        149           323              0
## 49    Publik         375         0        375           608              0
## 50    Publik         288       114        174           413              1
## 51    Publik         442       218        224           502              1
## 52    Publik         273       136        137           327              0
## 53    Publik         384       172        212           614              0
## 54    Publik         440       197        243           470              0
## 55    Publik         178        81         97           182              0
## 56    Publik         286       134        152           496              0
## 57    Publik         267       131        136           563              1
## 58    Publik         358       130        228           274              0
## 59    Privat         416       203        213           536              0
## 60    Publik         257       104        153           304              1
## 61    Publik         383       177        206           440              0
## 62    Publik         352       168        184           651              1
## 63    Publik         371        79        292           397              0
## 64    Publik         352       165        187           409              2
## 65    Publik         163        76         87           211              4
## 66    Privat         159        83         76           326              0
## 67    Publik         337       155        182           418              2
## 68    Publik         221       111        110           256              0
## 69    Publik         258       128        130           368              2
## 70    Publik         450         0        450             0              0
## 71    Privat         246       112        134           397              0
## 72    Privat         467       198        269           532              1
## 73    Publik         119        40         79           128              0
## 74    Publik         298       144        154           317              2
## 75    Privat         229       116        113           407              0
## 76    Publik         365       177        188           442             14
## 77    Publik         186        94         92           112              2
## 78    Publik         395       180        215           609              3
## 79    Publik         271       124        147           230              1
## 80    Publik         116        55         61           230              0
## 81    Publik         331       164        167           561              1
## 82    Publik         360       148        212           549              5
## 83    Publik         200       103         97           332              0
## 84    Publik         403       198        205           489              1
## 85    Publik         229        90        139           403              2
## 86    Publik         146        66         80           398              1
## 87    Publik         276       129        147           190              2
## 88    Publik         219       103        116           554              0
## 89    Publik         269       131        138           330              0
## 90    Publik         341       168        173           434              1
## 91    Publik          35        14         21            35              0
## 92    Publik         268       122        146           466              0
## 93    Publik         308       144        164           240              4
## 94    Publik         240       113        127           278              0
## 95    Publik         408         0        408           468              0
## 96    Publik         318       156        162           335              0
## 97    Publik         161        73         88           189              0
## 98    Publik         290       150        140           426              0
## 99    Publik         373       182        191           563              0
## 100   Publik         252       120        132           407              0
## 101   Publik         419       198        221           582              3
## 102   Publik         285       135        150           390              0
## 103   Publik         373       183        190           427              1
## 104   Publik         120        47         73           147              2
## 105   Publik         131        62         69           131              0
## 106   Publik         168        79         89           105              0
## 107   Publik         294       125        169           238              0
## 108   Publik         252       114        138           219              0
## 109   Publik         308       143        165           281              6
## 110   Publik         323       139        184           467              0
## 111   Publik         270       126        144           281              6
## 112   Publik         198        72        126           301              1
## 113   Publik         308       146        162           452              0
## 114   Publik         448       214        234           529              3
## 115   Publik         506       249        257           494              0
## 116   Publik         135        54         81           159              0
## 117   Publik         486       224        262           413              1
## 118   Publik         197        80        117           203              0
## 119   Publik          90        37         53           173              0
## 120   Publik         384         0        384           346              0
## 121   Publik         223       100        123           624              7
## 122   Publik         236       112        124           442              1
## 123   Privat         168        50        118           380              0
## 124   Publik         222        96        126           274              0
## 125   Publik         401       185        216           373              3
## 126   Privat         506       241        265           413              0
## 127   Publik         390       178        212           523              3
## 128   Publik         612       305        307           388              0
## 129   Publik         386       177        209           399              0
## 130   Publik         485       249        236           521              1
## 131   Publik         308       147        161           313              1
## 132   Publik         132        62         70            76              0
## 133   Publik          74        35         39           331              0
## 134   Publik         419       190        229           335              0
## 135   Publik         468       201        267           446              0
## 136   Publik         471       225        246           530              0
## 137   Publik         237        98        139           448              0
## 138   Publik          96        50         46           172              1
## 139   Publik         527       246        281             0              0
## 140   Publik         180        81         99           211              0
## 141   Publik         375       189        186           327              0
## 142   Publik         328       157        171           399              4
## 143   Publik         287       147        140           248              5
## 144   Publik         530         0        530           400              2
## 145   Publik          93        42         51           468              4
## 146   Publik         196        77        119           517              1
## 147   Publik         327       164        163           424              2
## 148   Publik         170        91         79            98              0
## 149   Publik         363       180        183           465              0
## 150   Publik         250       118        132           261              0
## 151   Publik         451       212        239           566              0
## 152   Publik         286       132        154           286              2
## 153   Publik         444       205        239           327              2
## 154   Publik         419       201        218           479              2
## 155   Publik         114        47         67           210              1
## 156   Privat         387       190        197           633              0
## 157   Publik         384       183        201           496              0
## 158   Publik         303       151        152           280              0
## 159   Publik         399       193        206           421              3
## 160   Publik         279       134        145           372              1
## 161   Publik         390       182        208           406              6
## 162   Publik         288       143        145           291              2
## 163   Publik         283       129        154             0              0
## 164   Publik         187        90         97           313              0
## 165   Publik         429       202        227           401              3
## 166   Publik         332       200        132           356              0
## 167   Publik         365       194        171           463              2
## 168   Publik         241       121        120           138              3
## 169   Privat         353       161        192           431              0
## 170   Publik         166        86         80           419              1
## 171   Publik         123        58         65           375              0
## 172   Publik         262       129        133           308              0
## 173   Privat         404       187        217           261              2
## 174   Privat         399       186        213             0              2
## 175   Publik         275       140        135           293              0
## 176   Publik         320       152        168           508              0
## 177   Publik         337       155        182           450              2
## 178   Publik         305       150        155           652              3
## 179   Privat         440       201        239           507              0
## 180   Publik         309         0        309           410              0
## 181   Publik         280       140        140           360              0
## 182   Publik         302       117        185           406              1
## 183   Publik         380       175        205           437              0
## 184   Publik         470       217        253           299              0
## 185   Publik         156        58         98           195              2
## 186   Publik         240       113        127           230              1
## 187   Publik          90        44         46           115              1
## 188   Publik         139        65         74           161              0
## 189   Privat         110        29         81           113              0
## 190   Publik         426       201        225           492              0
## 191   Publik         203        91        112           653              0
## 192   Publik         287       133        154           508              0
## 193   Publik         289       144        145           466              0
## 194   Publik         335       169        166           508              0
## 195   Publik         259       134        125           177              1
## 196   Publik         114        56         58           158              0
## 197   Publik         299       161        138           355              9
## 198   Publik         432       200        232           489              0
## 199   Privat         229       104        125           174              0
## 200   Publik         221        99        122           295              0
## 201   Privat         446       223        223           489              0
## 202   Publik         263       116        147           384              4
## 203   Publik         247       107        140           321              1
## 204   Publik         399         0        399           210              0
## 205   Privat         299       132        167           344              0
## 206   Publik         470       204        266           416              0
## 207   Publik         190        85        105           330              2
## 208   Publik         397       209        188           614              0
## 209   Publik         237       110        127           326              0
## 210   Publik         201       100        101           152              0
## 211   Publik         397       181        216           584              1
## 212   Privat         475       209        266           502              0
## 213   Publik         268       113        155           237              1
## 214   Publik         431         0        431           569              1
## 215   Publik         316       151        165           416              1
## 216   Publik         239         0        239           417              0
## 217   Publik         187        89         98           263              0
## 218   Publik         306       116        190           292              4
## 219   Publik         334       154        180           346              0
## 220   Publik         220        99        121           138              0
## 221   Publik         389       200        189           345              0
## 222   Publik         410       172        238           357              0
## 223   Publik         304       155        149           341              0
## 224   Publik         370       183        187           329              1
## 225   Publik         132        67         65           268              0
## 226   Publik         235         0        235           247              1
## 227   Publik         169        65        104           198              0
## 228   Publik         135        83         52             0              0
## 229   Publik         346       168        178           343              1
## 230   Publik         400       192        208             0              0
## 231   Publik         448       220        228           563              2
## 232   Publik         106        52         54           125              0
## 233   Publik         234        96        138           219              1
## 234   Publik         449       232        217           443              0
## 235   Privat         122        51         71           114              2
## 236   Publik         174        65        109            65              0
## 237   Publik         237        65        172           235              0
## 238   Privat         304         0        304           349              0
## 239   Privat         470       220        250           512              2
## 240   Publik         132        51         81           102              1
## 241   Publik         335       164        171           409              1
## 242   Publik         176        95         81           338              0
## 243   Publik         259         0        259           238              0
## 244   Publik         186         0        186           129              1
## 245   Publik         491       224        267           441              1
## 246   Publik         276       122        154           351              0
## 247   Privat         459       210        249           481              0
## 248   Publik         291       129        162           174              2
## 249   Publik         318       191        127           305              0
## 250   Publik         246       120        126           449              0
## 251   Publik         191       100         91           393              1
## 252   Publik         370       200        170           459              0
## 253   Publik         304       117        187           394              2
## 254   Privat         211       100        111           270              1
## 255   Publik         370       174        196           480              9
## 256   Publik         467       237        230             0              0
## 257   Publik         195        95        100           287              3
## 258   Publik         390       195        195           483              3
## 259   Publik         220        93        127           476              4
## 260   Publik         251       107        144           238              0
## 261   Publik         261       119        142           385              0
## 262   Publik         407       207        200           537              1
## 263   Publik         295       148        147           590              0
## 264   Publik         312       167        145           312              0
## 265   Publik         296       103        193           453              0
## 266   Publik         414       202        212           551              1
## 267   Publik         192        75        117           165              0
## 268   Publik         332       110        222           526              6
## 269   Publik         369       182        187           440              0
## 270   Publik         122        51         71           102              0
## 271   Publik         173        80         93           424              0
## 272   Publik         243       114        129           243              0
## 273   Publik         262       118        144           230              1
## 274   Publik         443       228        215           532              0
## 275   Publik         313       141        172           625              0
## 276   Publik         173        71        102           128              1
## 277   Publik         313       161        152           402              0
## 278   Publik         293       139        154           292              0
## 279   Publik         630       270        360           329              0
## 280   Publik         354       164        190           383              0
## 281   Publik         202        89        113           273              3
## 282   Publik         231       106        125           273              2
## 283   Publik         217       104        113           148              1
## 284   Publik         418       119        299           496              2
## 285   Publik         199        85        114           506              1
## 286   Publik         415       217        198           406              3
##     ballotsCast invalidVotes validVotes lsi  ps pkd sfida pr  pd pbdksh adk psd
## 1            95            0         95   2  78   0     0  0  15      0   0   0
## 2           369            4        365  98 182   0     0  1  79      0   0   1
## 3           516            5        511  96 282   0     4  0 101      0   0   0
## 4           446            9        437  50 202   0     2  1 138      0   0  35
## 5           223            7        216  21  67   1     2  0 121      0   0   0
## 6           387            9        378  37 130   0     1  0 110      0   0   1
## 7           436           11        425  38 238   1     0  1 118      0   0   5
## 8           487            9        478 160 267   0     0  0  41      0   0   0
## 9           287            5        282  14 161   0     0  1  72      0   0   0
## 10          319            3        316  46 193   0     0  1  74      0   0   1
## 11          310            3        307  40 133   0     9  0  96      0   0   0
## 12          290            4        286  30 183   1     0  1  69      0   0   0
## 13          350            7        343  52 154   2     0  0  59      0   0   3
## 14          236            0        236  18 106   0     0  2 101      0   0   0
## 15          529           10        519  44 352   1     0  3 114      0   0   0
## 16          393            5        388  92 186   2     0  1  99      0   0   0
## 17          305            5        300  80  82   0     0  0  57      0   0   0
## 18          246            6        240  45 151   0     0  0  32      0   1   0
## 19          470           11        459  40 254   0     3  0 121      0   0   0
## 20          329            4        325  45  91   1     0  0  68      0   0   0
## 21          321           11        310  39 191   0     0  0  66      0   0   1
## 22          364            4        360  75 157   0     0  1 111      0   0   0
## 23          265            2        263  19 159   1     0  0  80      0   0   0
## 24          255            5        250  91 126   0     0  0  30      0   0   0
## 25          286            4        282  14 215   0     0  1  40      0   0   0
## 26          443           12        431  36 208   0     0  2 180      1   0   0
## 27          362            7        355 133 154   0     2  0  62      0   1   0
## 28           61            2         59  27   6   0     0  1  25      0   0   0
## 29          248            7        241  30 156   0     1  0  46      1   0   0
## 30          275            7        268  24  72   0     0  0  52      0   0   0
## 31          302            5        297  31 178   0     0  0  82      0   0   0
## 32          440            9        431  86 185   0     0  2 141      0   0   1
## 33          280            7        273  66 186   0     0  0  20      0   0   0
## 34          446            6        440  43 255   0     2  1 108      0   0   0
## 35          237            9        228  63  75   0     0  0  27      0   0   0
## 36          268            5        263  51 114   1     0  1  69      1   0   0
## 37          266            6        260  13 160   0     0  0  77      0   0   2
## 38          473           10        463  61 283   0     0  1  82      1   0   0
## 39          198            5        193   7  36   0     0  1 147      0   0   0
## 40          348           10        338  44 199   1     2  0  68      0   0   0
## 41          468            7        461  43 262   0     7  1 107      0   0   0
## 42           40            0         40   8  19   1     0  0  12      0   0   0
## 43          335            7        328  44 154   0     2  1 110      0   0   0
## 44          375            9        366  50 192   1     1  1  98      0   0   0
## 45          301            3        298  65 146   0     1  3  76      0   0   0
## 46          223            3        220  26 161   0     0  1  30      0   0   0
## 47          112            1        111   5  32   0     0  0  63      0   0   0
## 48          265            4        261  73  43   0     0  1 137      0   0   0
## 49          375           10        365  38 218   1     1  0  84      0   0   3
## 50          288           11        277  29 170   0     0  0  65      0   0   0
## 51          442           20        422  38 248   0     3  0 118      0   0   0
## 52          273            3        270  28  65   0     0  0  88      0   0   0
## 53          384           12        372  56 245   0     0  1  45      0   0   0
## 54          440            6        434  25 246   0     0  1 131      0   0   0
## 55          178            2        176  41  90   0     0  0  41      0   0   0
## 56          286           11        275  74 109   1     0  0  87      0   0   0
## 57          267            2        265  49 137   0     0  1  71      0   0   0
## 58          358            9        349  43  79   0     0  1 124      0   0   0
## 59          416            8        408  44 169   2     0  1 176      1   0   0
## 60          257           11        246  12 118   2     0  0  71      0  34   3
## 61          383            2        381  58 170   0     1  3 132      0   0   0
## 62          352            7        345  82 145   0     0  0 105      0   0   0
## 63          371            4        367  52 114   0     1  0 156      0   4  24
## 64          352            5        347  53 208   0     0  0  75      0   0   0
## 65          163            2        161  40 104   0     0  0  13      0   0   1
## 66          159            3        156  21  52   0     0  1  68      0   0   4
## 67          337           10        327  49 136   3     0  1 129      0   0   0
## 68          221            3        218  27 134   0     0  1  48      0   0   0
## 69          258            5        253  34 127   0     0  0  86      0   0   0
## 70          450            6        444  56 163   0     0  2 147      1   0   0
## 71          246            6        240  37 103   0     0  0  87      0   0   2
## 72          467            8        459  42 173   0     1  2 118      0   0   0
## 73          119            0        119  12  61   0     0  0  46      0   0   0
## 74          298           11        285  49 119   0     0  0 109      2   0   0
## 75          229            5        223  87  70   0     0  0  38      0   0   0
## 76          365            7        358  16 150   0     0  0 170      0   0   0
## 77          186            2        184  25  80   0     3  1  74      0   0   0
## 78          395            9        386  53 226   0     0  0  71      0   0   0
## 79          271            3        268  18  93   0     0  0  23      0   0   0
## 80          116            0        116  16  86   0     0  0  12      0   0   0
## 81          331            7        324  33 176   0     1  1  96      0   0   0
## 82          360           13        347  11 253   0     1  0  77      0   0   1
## 83          200            4        196  13 119   4     4  0  45      0   0   0
## 84          403           10        393  51 193   0     3  0 113      0   0   0
## 85          229            4        225  62  69  12     0  0  43      0   0  39
## 86          146            3        143  14  92   2     0  0  29      0   0   0
## 87          276            3        273  41 152   0     0  1  76      0   0   0
## 88          219            3        216  54 118   0     0  0  43      0   0   0
## 89          269            8        261  46 103   1     1  0  69      0   0   0
## 90          341           10        331  33 174   0     1  0 112      0   0   0
## 91           35            0         35  20  12   0     0  0   1      0   0   0
## 92          268            8        260  43 126   0     0  0  70      1   0   0
## 93          308           11        297  86 174   0     0  1  34      0   0   0
## 94          240            3        237  19 167   0     0  0  50      0   0   0
## 95          408            6        402  53 197   2     0  0 136      4   0   0
## 96          318           12        306  28 164   0     0  0  96      0   2   0
## 97          161            2        159  21  89   1     0  0  48      0   0   0
## 98          290           12        278  12  98   0     0  1  97      0   2  50
## 99          373            6        367  48 160   0     0  1 135      0   0   0
## 100         252            0        252   5  31   0     0  0 123      1   0  82
## 101         419            7        412  10 140   1     0  1 170      0   1  81
## 102         285            3        282  50  98   0     0  1 112      0   0   0
## 103         373            5        368  49 165   1     3  1 133      0   0   0
## 104         120            2        118  34  78   0     0  0   5      0   0   0
## 105         131            3        128  14  47   0     0  1  25      0   0   0
## 106         168            3        165  19  99   0     0  2  45      0   0   0
## 107         294           10        284  35 147   0     2  0  80      0   0   0
## 108         252            5        247  58 124   0     0  0  51      0   0   0
## 109         308            2        306   8 130   1     0  0  35      0   0   0
## 110         323            4        319  61 162   0     0  0  77      0   0   0
## 111         270            8        262  13 167   0     0  0  79      0   0   0
## 112         198            4        194  58  91   0     0  0  43      1   0   1
## 113         308            5        303  36 183   0     0  1  70      1   0   0
## 114         448            7        441  47 248   1     0  1 118      0   0   1
## 115         506            8        498  66 256   0     3  1 143      0   0   0
## 116         135            0        135   9  49   0     0  0  75      0   0   0
## 117         486           11        475  37 279   0     1  1 133      0   0   1
## 118         197            5        192   7  55   0     0  0 128      0   0   0
## 119          90            0         90  11  66   0     0  0  11      0   0   0
## 120         384           10        374  43 225   1     0  0  99      0   0   0
## 121         223            7        213  18  84   3     0  1 101      0   0   0
## 122         236            4        232  17 105   0     1  1  92      2   0   0
## 123         168           10        158  13  85   0     1  0  55      0   0   0
## 124         222            3        219  58  63   8     0  0  70      0   0   6
## 125         401           13        388  58 187   1     0  1 127      0   0   0
## 126         506           19        487  93 332   1     0  0  49      0   0   1
## 127         390            3        387  58 254   0     0  1  42      0   0   0
## 128         612           16        596  47 293   0     0  3 237      1   0   0
## 129         386           14        372  49 186   0     3  1  85      0   0   0
## 130         485           10        475  53 228   1    12  1 144      0   0   0
## 131         308            4        304  25 134   0     1  0  66      0   0   0
## 132         132            4        128  19  30   0     0  0  10      0   0   0
## 133          74            0         74   2  50   0     0  0   6      0   0   0
## 134         419            8        411  97 123   2     0  0 164      1   2   0
## 135         468            1        467 138 260   0     0  0  64      0   0   0
## 136         471           10        461 104 256   0     0  1  94      0   0   0
## 137         237            3        234  25  83   1     1  0 100      0   0  11
## 138          96            4         92  19  40   0     0  0  32      0   0   1
## 139         527           13        514  84 290   0     0  0 128      0   0   0
## 140         180            6        174  12 113   0     0  0  34      0   0   1
## 141         375            5        370  81 170   0     7  1  76      0   0   7
## 142         328           11        317  82 127   0     0  0  77      0   0   0
## 143         287            6        281  16 104   0     0  0 161      0   0   0
## 144         530           13        517  22 399   1     0  1  32      0   0   0
## 145          93            0         93   3  32   0     0  0  14      0   0   0
## 146         196            6        190   4  89   1     0  3  87      0   0   0
## 147         327           11        316  36 168   0     9  0  75      0   0   0
## 148         170            2        168  31  48   0     0  0  89      0   0   0
## 149         363            7        356  42 201   0     2  0  82      0   0   3
## 150         250           15        235  49 100   0     0  1  83      0   0   0
## 151         451            2        449  22 238   0     0  4 156      0   0   0
## 152         286            4        282  27 172   1     0  0  67      0   0   0
## 153         444            8        436  59 231   0     0  0 108      0   0   0
## 154         419            8        411  47 146   0     0  1  89      0   0   0
## 155         114            2        112   1  34   0     0  0  67      1   0   9
## 156         387            4        383  50 253   0     0  0  63      0   0   0
## 157         384           10        374  15 117   4     0  3 222      0   0   0
## 158         303            4        299   5 103   0     0  0  71      0   0   1
## 159         399            9        390  37 246   2     0  0  86      0   0   0
## 160         279            2        277  69 154   0     0  0  54      0   0   0
## 161         390            9        381  61 176   5     6  0 121      0   0   0
## 162         288            6        282  24 126   0     2  1 119      0   0   0
## 163         283            7        276  47 121   3     0  0  83      1   0   9
## 164         187            1        186  10 133   0     0  0  33      0   0   0
## 165         429            7        422  48 189   0    12  0 118      0   0   0
## 166         332            6        326  26 202   0     0  0  91      0   0   0
## 167         365            4        361  48 177   0     7  0  96      0   0   0
## 168         241            5        236  17 143   0     0  1  72      0   0   0
## 169         353            1        352  96 168   2     1  0  66      1   1   0
## 170         166            4        162  29  48   2     0  0  70      0   0   8
## 171         123            2        121   8  85   2     0  0  22      0   0   0
## 172         262            7        255  73 161   0     0  0  20      0   0   0
## 173         404            7        397  26 310   0     1  0  46      1   0   0
## 174         399            4        395  57 204   1     0  0  96      0   0   0
## 175         275            5        270  22  61   1     0  0 159      1   0   0
## 176         320            1        319  34 202   1     0  0  73      0   0   0
## 177         337            6        331  42 173   0     1  0  85      0   0   0
## 178         305            4        301  14 176   0     0  0  71      0   0   0
## 179         440            6        434  79 152   1     0  2 193      0   0   0
## 180         309            6        303  48 127   1     0  0 110      0   0   0
## 181         280            2        278  31 149   0     3  0  81      0   0   0
## 182         300            6        294  23 131   0     1  2  86      0   0  15
## 183         380            4        376  29 225   0     0  0 115      0   0   1
## 184         470           18        452  73 226   1     1  1 134      1   0   0
## 185         156            0        156  25  65   0     0  0  55      3   0   0
## 186         240            0        240  33 157   0     0  0  42      0   0   0
## 187          90            2         88   0  74   1     0  0  12      0   0   0
## 188         139            0        139  13  81   0     0  0  40      0   0   0
## 189         110            0        110  24  53   0     0  0  31      0   0   0
## 190         426            9        417  91 280   0     0  2  40      0   0   0
## 191         203            1        202  10 164   0     0  0  27      0   0   0
## 192         287            4        283  17 201   0     0  0  43      0   0   0
## 193         289            6        283  38 156   0     0  1  69      0   0   0
## 194         335            7        328  45 210   0     4  0  57      0   1   0
## 195         259            7        252 153  87   0     0  0  12      0   0   0
## 196         114            2        112   6  36   0     0  1  11      0   0   0
## 197         299           12        287  32 151   0     6  0  76      0   0   0
## 198         432            2        430  31 250   0     0  0  97      0   0   0
## 199         229            2        227   5 161   0     0  0  43      0   0   0
## 200         221            4        217   8  97   0     0  0  67      0   0   1
## 201         446            5        441  41 103   0     0  0 153      0   1   2
## 202         263            0        263  16 209   0     0  0  38      0   0   0
## 203         247            1        246  44 139   0     0  0  51      0   0   0
## 204         399            8        391  23 146   1     4  0 212      0   0   0
## 205         299            4        295  42 146   1     0  0  83      0   0   1
## 206         470           11        459  51 161   2     0  2 215      0   0   1
## 207         190            1        189   9  79   0     0  1  93      1   0   0
## 208         397            9        388  58 126   0     0  2 186      0   1   2
## 209         237            3        234  25 105   0     0  1  69      0   0  27
## 210         201            0        201  48  94   0     0  0  38      0   0   0
## 211         397           10        387  43 162   0     1  1 155      2   0   0
## 212         475           15        460  48 155   1     0  1 242      1   0   1
## 213         268            5        263  28 158   1     0  0  61      0   0   0
## 214         431            4        427  71 227   1     0  0  66      0   0   0
## 215         316            6        310  15 143   2     0  1 121      2  21   2
## 216         239            1        238  33 123   7     2  0  61      0   0   0
## 217         187            0        187  13 133   0     0  1  37      0   0   0
## 218         306            4        302  91 146   0     0  0  58      0   0   0
## 219         334            0        334  29 134   0     0  0 120      0   0  25
## 220         220            1        219  54  83   0     0  0  32      0   0   0
## 221         389           17        372  62 204   0     0  0  68      0   0   0
## 222         410            7        403  75 203   1     0  1  90      0   0   1
## 223         304            1        303  33 149   0     0  2 102      0   0   0
## 224         370            5        365  63 196   0     6  0  75      0   0   2
## 225         132            0        132   5  58   0     0  2  61      0   0   0
## 226         235            6        229  23  85   0     0  1  91      0   0   0
## 227         169            0        169  11 121   0     0  0  11      0   0  26
## 228         135            2        133  45  61   1     0  0  23      1   0   0
## 229         346            7        339  57 165   1     0  0 112      0   0   1
## 230         400           12        388  33 186   1     0  2 148      5   0   0
## 231         448            9        439  60 198   0     1  1 109      0   0   0
## 232         106            5        101   9  36   0     0  0  55      0   0   0
## 233         234            7        227  59  76   0     0  2  86      0   0   0
## 234         449            7        442  34 252   0     1  0  99      0   0   0
## 235         122            0        122  17  71   0     0  0  32      0   0   0
## 236         174            1        173  47 103   0     0  0  23      0   0   0
## 237         237           22        215  21  32   0     0  0  56      0   0   0
## 238         304            4        300  40 158   0     0  0  91      1   0   0
## 239         470            9        461  24 240   0     1  6 179      0   0   1
## 240         132            2        130   0  33   0     0  0  79      0   0  15
## 241         335            2        333  27 224   0     0  0  82      0   0   0
## 242         176            2        174  25  82   0     1  0  40      0   0   0
## 243         259            0        259  53 126   0     0  2  77      0   0   0
## 244         186            2        184  74  77   1     0  0  15      0   0   0
## 245         491            8        483  75 224   0     1  1 142      0   0   0
## 246         276            5        271  51 145   1     1  1  71      0   0   0
## 247         459           19        440  72 266   0     2  0  83      1   0   0
## 248         291            4        287  48 134   0     0  1 103      0   0   0
## 249         318            8        310  23 190   0     0  0  85      0   0   0
## 250         246            1        245  35 130   0     1  0  63      0   0   0
## 251         191            3        188  47  81   0     0  0  50      0   0   0
## 252         370            9        361 112 199   0     0  0  44      1   0   0
## 253         304            7        297  25 150   2     1  0  89      1  12   8
## 254         211            9        202  44  54   0     0  1  93      0   0   4
## 255         370            5        365  18 185   0     0  1 141      0   0   0
## 256         467            9        458  89 167   1     3  0 168      9   0   1
## 257         195            5        190  11 113   1     1  0  59      0   0   0
## 258         390            6        384  40 206   0     0  2 128      0   0   0
## 259         220            1        219  23 139   0     0  0  54      0   0   0
## 260         251            6        245   4  48   0     0  0 165      0   0  24
## 261         261            4        257  16 174   0     0  2  47      0   0   0
## 262         407            5        402  32 184   0     0  2 134      0   0   0
## 263         295            4        291  37 139   0     0  0  82      0   0   0
## 264         312            7        305   3 125   1     0  1  37      0   0   0
## 265         296            9        287  10  51   2     0  2  41      1   0   1
## 266         414           13        401  69 201   1     0  0 121      1   0   0
## 267         192            0        192  12  69   0     0  0 111      0   0   0
## 268         332            8        324   9 182   0     0  0 129      0   0   0
## 269         369            9        360  28 195   0     0  0  98      0   0   1
## 270         122            6        116  31  65  19     0  1   0      0   0   0
## 271         173            4        169  33  43   0     0  0  61      0   0  26
## 272         243            9        234  24 129   0     0  0  77      0   0   0
## 273         262            4        257   7 141   0     0  1 103      0   0   0
## 274         443            3        440  87 250   0     0  2  84      0   0   0
## 275         313           21        292 128 114   1     0  0  43      1   0   1
## 276         173            3        170  38 102   0     1  0  25      0   0   0
## 277         313            5        308  31 175   0     0  1  79      0   0   4
## 278         293            2        291  33 133   0     0  1 124      0   0   0
## 279         630           15        615  56 482   0     0  0  61      0   0   0
## 280         354            5        349  42 188   0     1  0 106      0   0   0
## 281         202            2        200   4  64   0     0  0 114      0   0  11
## 282         231            0        231  61 125   0     0  0  39      0   0   0
## 283         217            3        214  11  31   0     0  0 118      0   0   0
## 284         418            6        412  58 163   0     0  2 160      5   0   0
## 285         199            3        196   9 111   0     0  1  70      0   0   0
## 286         415            9        406  56 186   0     2  0 127      0   0   0
##     ad frd pds pdiu aak mega pksh apd libra psSeats pdSeats lsiSeats pdiuSeats
## 1    0   0   0    0   0    0    0   0     0       8       3        1         0
## 2    0   2   0    0   0    1    0   0     1       4       1        2         0
## 3    0   3   2    5   0    0    0   0    18      18      11        5         0
## 4    0   1   1    4   0    0    1   1     1       8       4        2         0
## 5    0   0   0    0   0    0    2   0     2       3       3        1         0
## 6    0   0   0   91   0    0    2   0     6       7       3        2         2
## 7    0   2   2   10   0    0    0   0    10      18      11        5         0
## 8    0   1   2    2   0    0    0   0     5       4       1        2         0
## 9    0   0   0   31   0    0    0   0     3       8       3        1         0
## 10   0   0   1    0   0    0    0   0     0       6       4        1         0
## 11   0   1   0   10   0    0    0   0    18      18      11        5         0
## 12   0   0   1    1   0    0    0   0     0      10       4        2         0
## 13   0   1   1   70   1    0    0   0     0      18      11        5         0
## 14   0   0   0    7   0    0    0   0     2       3       3        1         0
## 15   0   0   0    5   0    0    0   0     0      10       4        2         0
## 16   0   6   0    0   0    0    0   0     2       8       4        2         0
## 17   0   0   0   81   0    0    0   0     0       7       3        2         2
## 18   0   0   0    4   0    0    0   0     7       4       1        2         0
## 19   0   2   1   14   0    0    0   1    23      18      11        5         0
## 20   0   4   0  105   0    0    2   0     9       7       3        2         2
## 21   0   1   0    8   0    0    0   0     4      18      11        5         0
## 22   0   0   0   11   0    0    0   2     3       3       3        1         0
## 23   0   0   0    2   0    0    0   0     2      18      11        5         0
## 24   0   0   0    0   0    0    0   0     3       6       4        1         0
## 25   0   5   1    4   0    0    0   0     2       8       3        1         0
## 26   0   2   0    0   1    0    0   0     1       3       1        1         0
## 27   0   0   0    0   0    0    0   1     2       4       1        2         0
## 28   0   0   0    0   0    0    0   0     0       4       1        2         0
## 29   0   1   0    0   0    0    0   0     6      10       4        2         0
## 30   0   0   0  117   0    0    0   0     3       7       3        2         2
## 31   0   0   0    5   0    0    0   0     1      10       4        2         0
## 32   0   0   0   13   0    0    0   0     3      10       4        2         0
## 33   0   0   0    0   1    0    0   0     0       4       1        2         0
## 34   0   1   0   20   0    0    0   0    10      18      11        5         0
## 35   0   0   0   60   0    0    1   0     2       7       3        2         2
## 36   0   4   0   16   0    0    0   1     5       8       4        2         0
## 37   0   0   0    6   0    0    0   0     2       8       4        2         0
## 38   0   0   1   34   0    0    0   0     0       7       3        2         2
## 39   0   0   0    0   0    0    0   0     2       1       2        0         0
## 40   0   1   1    9   0    0    0   0    13      18      11        5         0
## 41   0   2   1   16   0    1    0   0    21      18      11        5         0
## 42   0   0   0    0   0    0    0   0     0       3       1        1         0
## 43   0   2   0    6   2    0    1   0     6       8       4        2         0
## 44   0   1   0    7   0    0    0   0    15      18      11        5         0
## 45   0   0   0    5   1    0    0   0     1       4       1        2         0
## 46   0   0   0    1   0    0    0   0     1       8       3        1         0
## 47   0   0   0    0   0   11    0   0     0       8       3        1         0
## 48   0   0   0    0   0    0    1   0     6       6       4        1         0
## 49   0   0   1    3   0    0    0   0    16      18      11        5         0
## 50   0   1   0   12   0    0    0   0     0       3       3        1         0
## 51   0   1   0    3   0    2    0   0     9       6       4        1         0
## 52   0   0   0   89   0    0    0   0     0       7       3        2         2
## 53   0   0   0   22   0    1    0   0     2       8       3        1         0
## 54   0   0   0    1   0    0    0   0    30      10       4        2         0
## 55   0   0   0    0   0    0    0   0     4       6       4        1         0
## 56   0   0   0    2   0    1    0   0     1       3       1        1         0
## 57   0   0   0    6   0    0    0   0     1       4       1        2         0
## 58   0   0   2   96   0    0    0   1     3       2       2        1         1
## 59   0   2   0    6   0    0    0   2     5      18      11        5         0
## 60   0   2   0    1   0    0    1   0     2       8       4        2         0
## 61   2   0   0   11   0    0    0   0     4      18      11        5         0
## 62   0   0   0   13   0    0    0   0     0      10       4        2         0
## 63   0   2   0   11   0    0    0   0     3       4       5        1         0
## 64   0   0   0   11   0    0    0   0     0       4       1        2         0
## 65   0   0   1    0   0    0    0   1     1       3       1        1         0
## 66   0   0   0    6   0    0    0   0     4       4       5        1         0
## 67   0   3   0    4   2    0    0   0     0      18      11        5         0
## 68   1   0   0    1   0    0    0   0     6      10       4        2         0
## 69   0   0   0    4   0    0    0   0     2      10       4        2         0
## 70   0   4   1   67   0    0    0   0     3       7       3        2         2
## 71   0   0   0    5   0    0    0   0     6      18      11        5         0
## 72   3   5   0  106   0    0    0   0     9       7       3        2         2
## 73   0   0   0    0   0    0    0   0     0       4       5        1         0
## 74   0   1   1    3   0    0    0   0     1       2       2        1         1
## 75   0   0   0   27   0    0    0   0     1       7       3        2         2
## 76   0   0   0   20   0    0    0   0     2       6       4        1         0
## 77   0   0   0    0   0    0    0   0     1       6       4        1         0
## 78   0   4   0   23   0    7    0   0     2       8       3        1         0
## 79   0  11   1  120   0    0    1   0     1       7       3        2         2
## 80   0   0   0    0   0    0    0   0     2      10       4        2         0
## 81   0   0   0    5   0    0    0   0    12      18      11        5         0
## 82   0   0   0    2   0    0    0   0     2       8       3        1         0
## 83   0   0   0    2   0    0    1   0     8       6       4        1         0
## 84   0   1   2   16   0    0    0   0    14      18      11        5         0
## 85   0   0   0    0   0    0    0   0     0       4       5        1         0
## 86   0   0   0    0   0    2    0   0     4       6       4        1         0
## 87   0   0   0    2   0    0    0   0     1       3       3        1         0
## 88   0   0   0    0   0    0    0   0     1       4       1        2         0
## 89   0   1   5   32   0    0    0   0     3      18      11        5         0
## 90   0   0   1    2   0    1    0   0     7       6       4        1         0
## 91   0   0   0    0   0    2    0   0     0       8       3        1         0
## 92   0   0   0   10   0    0    0   0    10       8       4        2         0
## 93   0   0   0    1   0    0    0   0     1       4       1        2         0
## 94   0   0   0    0   0    0    1   0     0       6       4        1         0
## 95   0   0   0    5   0    0    0   0     5       3       3        1         0
## 96   0   0   0    8   1    0    0   0     7      18      11        5         0
## 97   0   0   0    0   0    0    0   0     0       3       1        1         0
## 98   0   0   1   16   0    0    0   0     1       4       5        1         0
## 99   0   0   3    2   0    0    0   0    18       6       4        1         0
## 100  0   0   2    6   0    0    1   0     1       4       5        1         0
## 101  0   0   0    4   0    0    0   0     4       4       5        1         0
## 102  0   0   1   15   0    0    0   0     5       3       3        1         0
## 103  0   0   0   14   0    0    1   0     1       8       4        2         0
## 104  0   0   0    0   0    0    0   0     1       4       1        2         0
## 105  0   0   0   41   0    0    0   0     0       7       3        2         2
## 106  0   0   0    0   0    0    0   0     0      18      11        5         0
## 107  0  13   0    6   0    0    0   0     1      18      11        5         0
## 108  0   6   2    6   0    0    0   0     0      10       4        2         0
## 109  0   0   1  128   0    2    0   0     1       2       2        1         1
## 110  0   3   0   10   0    0    1   0     5      10       4        2         0
## 111  0   0   0    3   0    0    0   0     0      10       4        2         0
## 112  0   0   0    0   0    0    0   0     0       4       1        2         0
## 113  0   0   0    0   0    0    1   0    11      18      11        5         0
## 114  0   1   0   11   0    0    0   8     5      18      11        5         0
## 115  0   0   0   14   0    0    1   0    14      18      11        5         0
## 116  0   0   0    0   0    0    0   0     2       1       2        0         0
## 117  0   1   2    9   0    0    1   0    10      18      11        5         0
## 118  0   0   0    0   0    0    0   0     2       1       2        0         0
## 119  0   2   0    0   0    0    0   0     0       4       1        2         0
## 120  0   0   0    2   0    0    0   0     4       8       4        2         0
## 121  0   0   0    6   0    0    0   0     0      18      11        5         0
## 122  0   0   0    9   0    0    0   4     1       3       3        1         0
## 123  0   0   1    0   0    2    0   0     1       8       3        1         0
## 124  0   0   0   12   0    0    0   1     1       4       5        1         0
## 125  0   0   0   11   0    0    1   0     2       2       2        1         1
## 126  0   0   1    4   0    0    0   0     6      10       4        2         0
## 127  0   0   2   21   0    1    0   0     8      10       4        2         0
## 128  1   0   0    0   0    0    0   0    14       1       2        0         0
## 129  0   3   0   29   0    0    0   0    16      18      11        5         0
## 130  1   3   0   12   0    0    0   0    20      18      11        5         0
## 131  0  13   0   62   0    0    0   0     3       7       3        2         2
## 132  0   0   0   69   0    0    0   0     0       7       3        2         2
## 133  0   0   0    0   0   16    0   0     0       8       3        1         0
## 134  0   1   1    6   0    0    0   2    12      18      11        5         0
## 135  0   0   0    0   0    0    5   0     0       8       3        1         0
## 136  1   0   0    0   0    0    0   0     5       3       1        1         0
## 137  0   0   0    9   0    0    0   0     4       4       5        1         0
## 138  0   0   0    0   0    0    0   0     0       4       1        2         0
## 139  0   0   3    0   0    0    0   0     9       3       1        1         0
## 140  0   0   0   11   0    0    0   0     3       3       3        1         0
## 141  0   0   0   12   0    0    1   1    14      18      11        5         0
## 142  0   0   1   21   0    0    0   1     8       8       4        2         0
## 143  0   0   0    0   0    0    0   0     0       1       2        0         0
## 144  0   1   0   58   0    0    0   0     3       7       3        2         2
## 145  0   0   0    0   0   44    0   0     0       8       3        1         0
## 146  0   0   0    4   0    0    0   0     2       8       3        1         0
## 147  0   5   3   10   2    0    0   0     8      18      11        5         0
## 148  0   0   0    0   0    0    0   0     0       1       2        0         0
## 149  0   0   1   17   0    0    0   0     8      18      11        5         0
## 150  0   0   0    0   0    0    0   0     2       6       4        1         0
## 151  0   7   0    8   0    0    1   0    13       8       4        2         0
## 152  2   1   0    0   1    0    0   6     5      18      11        5         0
## 153  0   0   0   15   0    0    0   0    23      18      11        5         0
## 154  0   6   1  119   0    1    0   0     1       7       3        2         2
## 155  0   0   0    0   0    0    0   0     0       4       5        1         0
## 156  0   0   0   11   0    0    1   0     5       8       3        1         0
## 157  0   0   0    5   0    0    0   5     3      18      11        5         0
## 158  0   0   0  117   0    0    0   1     1       7       3        2         2
## 159  0   0   0    5   0    0    2   0    12      18      11        5         0
## 160  0   0   0    0   0    0    0   0     0       3       1        1         0
## 161  0   1   0    6   0    0    1   0     4       8       4        2         0
## 162  0   0   0    3   0    0    0   2     5      18      11        5         0
## 163  0   0   0    2   0    0    0   0    10      18      11        5         0
## 164  0   1   1    4   0    0    0   0     4       8       3        1         0
## 165  0  10   0   29   0    0    1   2    13      18      11        5         0
## 166  0   0   2    2   0    0    0   0     3      10       4        2         0
## 167  0   3   0   12   0    1    0   0    17      18      11        5         0
## 168  0   0   0    0   0    0    0   0     3      18      11        5         0
## 169  0   0   0    8   0    0    0   0     9       8       4        2         0
## 170  0   0   0    0   0    0    0   0     5       4       5        1         0
## 171  0   0   0    1   0    0    0   0     3       8       3        1         0
## 172  0   0   0    0   0    0    0   0     1       4       1        2         0
## 173  0   0   1    2   0    0    0   0    10      10       4        2         0
## 174  0   4   0   21   0    1    0   0    11       8       4        2         0
## 175  1   2   1   22   0    0    0   0     0       8       4        2         0
## 176  1   1   0    2   0    0    0   1     4       8       3        1         0
## 177  0   0   0   28   0    0    0   0     2       4       1        2         0
## 178  0   0   0    0   0   40    0   0     0       8       3        1         0
## 179  0   0   4    0   0    0    0   1     2      18      11        5         0
## 180  1   0   0    2   0    0    3   2     9      18      11        5         0
## 181  1   5   1    3   0    0    0   0     4      18      11        5         0
## 182  0   2   0   33   0    0    0   0     1       4       5        1         0
## 183  0   0   0    4   0    0    1   0     1      10       4        2         0
## 184  0   2   0   12   0    0    1   0     0      18      11        5         0
## 185  0   0   0    5   0    0    0   0     3       3       3        1         0
## 186  0   0   0    5   0    0    0   0     3       3       1        1         0
## 187  0   0   0    0   0    0    0   0     1       4       5        1         0
## 188  0   1   0    3   0    0    0   1     0       3       3        1         0
## 189  0   1   0    1   0    0    0   0     0       4       1        2         0
## 190  0   0   1    3   0    0    0   0     0       4       1        2         0
## 191  0   0   0    0   0    0    1   0     0       8       3        1         0
## 192  0   0   0   15   0    0    0   0     7       8       3        1         0
## 193  0   0   3    3   0    0    0   1    12       6       4        1         0
## 194  0   0   0    5   0    2    0   0     4       6       4        1         0
## 195  0   0   0    0   0    0    0   0     0       4       1        2         0
## 196  0   0   1   57   0    0    0   0     0       7       3        2         2
## 197  0   1   1   14   0    0    0   0     6      18      11        5         0
## 198  0   6   0   43   0    3    0   0     0       8       3        1         0
## 199  0   0   0   16   0    0    0   0     2      10       4        2         0
## 200  0   1   0   43   0    0    0   0     0       8       3        1         0
## 201  0   0   1  135   0    0    0   0     5       7       3        2         2
## 202  0   0   0    0   0    0    0   0     0       8       3        1         0
## 203  0   1   0    2   0    0    0   0     9       4       1        2         0
## 204  0   0   0    0   0    0    0   1     4       6       4        1         0
## 205  0   0   0   20   0    0    0   0     2      10       4        2         0
## 206  0   7   0   17   0    0    0   0     3       8       4        2         0
## 207  1   0   0    0   0    0    0   0     5       6       4        1         0
## 208  0   0   0    1   0    0    0   1    11       6       4        1         0
## 209  0   0   0    7   0    0    0   0     0       4       5        1         0
## 210  0   0   0   20   0    0    0   0     1       7       3        2         2
## 211  0   0   0   17   0    0    0   0     6      10       4        2         0
## 212  0   2   0    0   0    0    0   0     9      18      11        5         0
## 213  0   0   0    6   0    0    0   9     0       3       1        1         0
## 214  0   5   0   50   0    0    0   0     7       7       3        2         2
## 215  0   0   0    3   0    0    0   0     0       8       4        2         0
## 216  0   0   0    8   0    0    0   0     4       8       4        2         0
## 217  0   0   0    3   0    0    0   0     0       8       3        1         0
## 218  0   0   0    6   0    0    0   0     1      18      11        5         0
## 219  0   0   0   25   0    0    1   0     0       4       5        1         0
## 220  0   0   0   50   0    0    0   0     0       7       3        2         2
## 221  0   0   0   33   0    0    2   1     2       7       3        2         2
## 222  0  17   0    4   0    0    0   1    10      18      11        5         0
## 223  4  10   0    2   0    0    0   0     1       8       4        2         0
## 224  0   2   0   11   0    0    0   0    10      18      11        5         0
## 225  0   1   0    5   0    0    0   0     0       8       4        2         0
## 226  0   0   0   29   0    0    0   0     0       2       2        1         1
## 227  0   0   0    0   0    0    0   0     0       4       5        1         0
## 228  0   0   0    2   0    0    0   0     0       3       3        1         0
## 229  0   2   0    0   0    0    0   0     1      10       4        2         0
## 230  0   0   0    8   0    0    2   1     2       3       3        1         0
## 231  0   8   1   57   0    0    0   0     4       7       3        2         2
## 232  0   0   0    0   0    0    0   0     1       1       2        0         0
## 233  0   0   0    1   0    0    0   0     3       6       4        1         0
## 234  0   2   0   43   0    0    0   0    11       7       3        2         2
## 235  0   0   0    2   0    0    0   0     0      10       4        2         0
## 236  0   0   0    0   0    0    0   0     0      18      11        5         0
## 237  0   0   1  105   0    0    0   0     0       7       3        2         2
## 238  0   0   1    6   0    0    0   0     3      10       4        2         0
## 239  0   0   0    8   0    0    0   0     2      18      11        5         0
## 240  0   0   0    1   0    0    0   0     2       4       5        1         0
## 241  0   0   0    0   0    0    0   0     0      10       4        2         0
## 242  0   6   0   19   0    0    0   0     1       8       4        2         0
## 243  0   0   0    0   0    0    0   0     1       3       1        1         0
## 244  0   0   0   17   0    0    0   0     0       7       3        2         2
## 245  0   0   6   11   0    0    0   4    19      18      11        5         0
## 246  0   0   0    0   0    0    0   0     1       6       4        1         0
## 247  0   1   1    6   0    0    0   1     7      18      11        5         0
## 248  0   1   0    0   0    0    0   0     0       4       1        2         0
## 249  0   6   0    4   0    0    2   0     0      10       4        2         0
## 250  0   1   0    6   0    0    0   0     9      18      11        5         0
## 251  0   0   0    6   0    0    0   0     4       8       4        2         0
## 252  1   1   0    3   0    0    0   0     0       7       3        2         2
## 253  0   1   0    3   0    0    1   0     4       8       4        2         0
## 254  1   0   0    4   0    0    0   0     1       4       5        1         0
## 255  0   1   0    4   0    0    0   0    15      18      11        5         0
## 256  0   0   1   11   0    0    0   1     7       3       3        1         0
## 257  0   0   0    5   0    0    0   0     0      10       4        2         0
## 258  0   0   0    5   0    0    0   0     3      10       4        2         0
## 259  0   0   0    2   0    0    0   0     1       8       3        1         0
## 260  0   0   0    4   0    0    0   0     0       4       5        1         0
## 261  0   0   0   18   0    0    0   0     0      10       4        2         0
## 262  5   0   0   44   0    0    0   0     1       7       3        2         2
## 263  0   0   0   22   0    0    1   0    10      18      11        5         0
## 264  0   0   0  137   0    0    1   0     0       7       3        2         2
## 265  0   0   3  176   0    0    0   0     0       7       3        2         2
## 266  0   0   0    3   0    0    0   0     5       8       4        2         0
## 267  0   0   0    0   0    0    0   0     0       1       2        0         0
## 268  0   0   0    4   0    0    0   0     0      10       4        2         0
## 269  0   0   0   26   0    0    0   0    12      18      11        5         0
## 270  0   0   0    0   0    0    0   0     0      10       4        2         0
## 271  0   0   0    5   0    0    0   0     1       4       5        1         0
## 272  0   0   0    0   0    0    0   0     4       6       4        1         0
## 273  1   2   0    0   0    0    1   0     1       6       4        1         0
## 274  0   0   0   10   0    0    0   0     7       4       1        2         0
## 275  0   0   0    1   0    3    0   0     0       3       1        1         0
## 276  0   0   0    1   0    0    0   0     3      18      11        5         0
## 277  0   3   0    5   0    0    0   0    10      18      11        5         0
## 278  0   0   0    0   0    0    0   0     0      10       4        2         0
## 279  0   0   0   12   0    0    0   0     4      18      11        5         0
## 280  0   1   0    6   0    0    1   0     4       6       4        1         0
## 281  0   0   0    5   0    0    0   0     2       4       5        1         0
## 282  0   0   0    5   0    0    0   0     1       6       4        1         0
## 283  0   0   1   52   0    0    0   0     1       2       2        1         1
## 284  0   0   0   12   0    0    0   0    12       3       3        1         0
## 285  0   0   3    0   0    2    0   0     0       8       3        1         0
## 286  0   2   0   12   0    0    0   4    17      18      11        5         0
##     psdSeats
## 1          0
## 2          0
## 3          0
## 4          0
## 5          0
## 6          0
## 7          0
## 8          0
## 9          0
## 10         0
## 11         0
## 12         0
## 13         0
## 14         0
## 15         0
## 16         0
## 17         0
## 18         0
## 19         0
## 20         0
## 21         0
## 22         0
## 23         0
## 24         0
## 25         0
## 26         0
## 27         0
## 28         0
## 29         0
## 30         0
## 31         0
## 32         0
## 33         0
## 34         0
## 35         0
## 36         0
## 37         0
## 38         0
## 39         0
## 40         0
## 41         0
## 42         0
## 43         0
## 44         0
## 45         0
## 46         0
## 47         0
## 48         0
## 49         0
## 50         0
## 51         0
## 52         0
## 53         0
## 54         0
## 55         0
## 56         0
## 57         0
## 58         0
## 59         0
## 60         0
## 61         0
## 62         0
## 63         1
## 64         0
## 65         0
## 66         1
## 67         0
## 68         0
## 69         0
## 70         0
## 71         0
## 72         0
## 73         1
## 74         0
## 75         0
## 76         0
## 77         0
## 78         0
## 79         0
## 80         0
## 81         0
## 82         0
## 83         0
## 84         0
## 85         1
## 86         0
## 87         0
## 88         0
## 89         0
## 90         0
## 91         0
## 92         0
## 93         0
## 94         0
## 95         0
## 96         0
## 97         0
## 98         1
## 99         0
## 100        1
## 101        1
## 102        0
## 103        0
## 104        0
## 105        0
## 106        0
## 107        0
## 108        0
## 109        0
## 110        0
## 111        0
## 112        0
## 113        0
## 114        0
## 115        0
## 116        0
## 117        0
## 118        0
## 119        0
## 120        0
## 121        0
## 122        0
## 123        0
## 124        1
## 125        0
## 126        0
## 127        0
## 128        0
## 129        0
## 130        0
## 131        0
## 132        0
## 133        0
## 134        0
## 135        0
## 136        0
## 137        1
## 138        0
## 139        0
## 140        0
## 141        0
## 142        0
## 143        0
## 144        0
## 145        0
## 146        0
## 147        0
## 148        0
## 149        0
## 150        0
## 151        0
## 152        0
## 153        0
## 154        0
## 155        1
## 156        0
## 157        0
## 158        0
## 159        0
## 160        0
## 161        0
## 162        0
## 163        0
## 164        0
## 165        0
## 166        0
## 167        0
## 168        0
## 169        0
## 170        1
## 171        0
## 172        0
## 173        0
## 174        0
## 175        0
## 176        0
## 177        0
## 178        0
## 179        0
## 180        0
## 181        0
## 182        1
## 183        0
## 184        0
## 185        0
## 186        0
## 187        1
## 188        0
## 189        0
## 190        0
## 191        0
## 192        0
## 193        0
## 194        0
## 195        0
## 196        0
## 197        0
## 198        0
## 199        0
## 200        0
## 201        0
## 202        0
## 203        0
## 204        0
## 205        0
## 206        0
## 207        0
## 208        0
## 209        1
## 210        0
## 211        0
## 212        0
## 213        0
## 214        0
## 215        0
## 216        0
## 217        0
## 218        0
## 219        1
## 220        0
## 221        0
## 222        0
## 223        0
## 224        0
## 225        0
## 226        0
## 227        1
## 228        0
## 229        0
## 230        0
## 231        0
## 232        0
## 233        0
## 234        0
## 235        0
## 236        0
## 237        0
## 238        0
## 239        0
## 240        1
## 241        0
## 242        0
## 243        0
## 244        0
## 245        0
## 246        0
## 247        0
## 248        0
## 249        0
## 250        0
## 251        0
## 252        0
## 253        0
## 254        1
## 255        0
## 256        0
## 257        0
## 258        0
## 259        0
## 260        1
## 261        0
## 262        0
## 263        0
## 264        0
## 265        0
## 266        0
## 267        0
## 268        0
## 269        0
## 270        0
## 271        1
## 272        0
## 273        0
## 274        0
## 275        0
## 276        0
## 277        0
## 278        0
## 279        0
## 280        0
## 281        1
## 282        0
## 283        0
## 284        0
## 285        0
## 286        0
```

Identify missing points between sample and collected data


```r
alsample <- rsamp(df = albania, 544)
alreceived <- rsamp(df = alsample, 390)
rmissing(sampdf = alsample,
         colldf = alreceived,
         col_name = qvKod)
```

## Stratified Sampling

A stratum is a subset of the population that has at least one common characteristic.

Steps:

1.  Identify relevant stratums and their representation in the population.
2.  Randomly sample to select a sufficient number of subjects from each stratum.

Stratified sampling reduces sampling error.


```r
library(dplyr)
# by number of rows
sample_iris <- iris %>%
    group_by(Species) %>%
    sample_n(5)
sample_iris
```

```
## # A tibble: 15 x 5
## # Groups:   Species [3]
##    Sepal.Length Sepal.Width Petal.Length Petal.Width Species   
##           <dbl>       <dbl>        <dbl>       <dbl> <fct>     
##  1          5           3.4          1.6         0.4 setosa    
##  2          4.5         2.3          1.3         0.3 setosa    
##  3          5.2         3.4          1.4         0.2 setosa    
##  4          4.9         3            1.4         0.2 setosa    
##  5          5.5         3.5          1.3         0.2 setosa    
##  6          5.5         2.4          3.8         1.1 versicolor
##  7          6.1         2.9          4.7         1.4 versicolor
##  8          6.4         2.9          4.3         1.3 versicolor
##  9          5.9         3.2          4.8         1.8 versicolor
## 10          6.3         2.5          4.9         1.5 versicolor
## 11          7.6         3            6.6         2.1 virginica 
## 12          6.5         3            5.2         2   virginica 
## 13          6.9         3.2          5.7         2.3 virginica 
## 14          6.5         3            5.8         2.2 virginica 
## 15          6.7         3.3          5.7         2.1 virginica
```

```r
# by fraction
sample_iris <- iris %>%
    group_by(Species) %>%
    sample_frac(size = .15)
sample_iris
```

```
## # A tibble: 24 x 5
## # Groups:   Species [3]
##    Sepal.Length Sepal.Width Petal.Length Petal.Width Species   
##           <dbl>       <dbl>        <dbl>       <dbl> <fct>     
##  1          5.4         3.9          1.3         0.4 setosa    
##  2          5.2         4.1          1.5         0.1 setosa    
##  3          5.1         3.4          1.5         0.2 setosa    
##  4          5.7         3.8          1.7         0.3 setosa    
##  5          5           3.3          1.4         0.2 setosa    
##  6          5.1         3.8          1.5         0.3 setosa    
##  7          4.6         3.6          1           0.2 setosa    
##  8          5.1         3.8          1.9         0.4 setosa    
##  9          6.2         2.9          4.3         1.3 versicolor
## 10          5.1         2.5          3           1.1 versicolor
## # ... with 14 more rows
```


```r
library(sampler)
# Stratified sample using proportional allocation without replacement
ssamp(df=albania, n=360, strata=qarku, over=0.1)
```

```
## # A tibble: 395 x 45
##    qarku  Q_ID bashkia BAS_ID zaz    njesiaAdministrative COM_ID qvKod  zgjedhes
##    <fct> <int> <fct>    <int> <fct>  <fct>                 <int> <fct>     <int>
##  1 Berat     1 Berat       11 ZAZ 64 "Berat "               1101 "\"33~      753
##  2 Berat     1 Berat       11 ZAZ 64 "Velabisht"            1111 "\"33~      757
##  3 Berat     1 Skrapar     17 ZAZ 66 "Potom"                1307 "\"35~      247
##  4 Berat     1 Kucove      13 ZAZ 63 "Kucove"               1201 "\"35~      589
##  5 Berat     1 Polican     15 ZAZ 65 "Terpan"               1109 "\"33~      523
##  6 Berat     1 Berat       11 ZAZ 64 "Roshnik"              1107 "\"34~      783
##  7 Berat     1 Kucove      13 ZAZ 63 "Kozare"               1202 "\"34~      594
##  8 Berat     1 Polican     15 ZAZ 65 "Vertop"               1112 "\"34~      593
##  9 Berat     1 Berat       11 ZAZ 64 "Sinje"                1108 "\"33~      647
## 10 Berat     1 Polican     15 ZAZ 65 "Polican"              1306 "\"35~      622
## # ... with 385 more rows, and 36 more variables: meshkuj <int>, femra <int>,
## #   totalSeats <int>, vendndodhja <fct>, ambienti <fct>, totalVoters <int>,
## #   femVoters <int>, maleVoters <int>, unusedBallots <int>,
## #   damagedBallots <int>, ballotsCast <int>, invalidVotes <int>,
## #   validVotes <int>, lsi <int>, ps <int>, pkd <int>, sfida <int>, pr <int>,
## #   pd <int>, pbdksh <int>, adk <int>, psd <int>, ad <int>, frd <int>,
## #   pds <int>, pdiu <int>, aak <int>, mega <int>, pksh <int>, apd <int>, ...
```

Identify number of missing points by strata between sample and collected data


```r
alsample <- rsamp(df = albania, 544)
alreceived <- rsamp(df = alsample, 390)
smissing(
    sampdf = alsample,
    colldf = alreceived,
    strata = qarku,
    col_name = qvKod
)
```

## Unequal Probability Sampling


```r
UPbrewer()
UPmaxentropy()
UPmidzuno()
UPmidzunopi2()
UPmultinomial()
UPpivotal()
UPrandompivotal()
UPpoisson()
UPsampford()
UPsystematic()
UPrandomsystematic()
UPsystematicpi2()
UPtille()
UPtillepi2()
```

## Balanced Sampling

-   Purpose: to get the same means in the population and the sample for all the auxiliary variables

-   Balanced sampling is different from purposive selection

Balancing equations

$$
\sum_{k \in S} \frac{\mathbf{x}_k}{\pi_k} = \sum_{k \in U} \mathbf{x}_k
$$

where $\mathbf{x}_k$ is a vector of auxiliary variables

### Cube

-   flight phase

-   landing phase


```r
samplecube()
fastflightcube()
landingcube()
```

### Stratification

-   Try to replicate the population based on the original multivariate histogram


```r
library(survey)
data("api")
srs_design <- svydesign(data = apistrat,
                        weights = ~pw, 
                        fpc = ~fpc, 
                        strata = ~stype,
                        id = ~1)
```


```r
balancedstratification()
```

### Cluster


```r
library(survey)
data("api")
srs_design <- svydesign(data = apiclus1,
                        weights = ~pw, 
                        fpc = ~fpc, 
                        id = ~dnum)
```


```r
balancedcluster()
```

### Two-stage


```r
library(survey)
data("api")
srs_design <- svydesign(data = apiclus2, 
                        fpc = ~fpc1 + fpc2, 
                        id = ~ dnum + snum)
```


```r
balancedtwostage()
```
