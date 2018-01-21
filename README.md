# About

This is a Python implementation of the SLOPE algorithm [1] for high-dimensional linear regression.
This software is compatible with linear model classes in scikit-learn.

## References

[1] M. Bogdan, E. van den Berg, C. Sabatti, W. Su, and E. J. Candès.
SLOPE—Adaptive variable selection via convex optimization.
The Annals of Applied Statistics, 9(3):1103--1140, 2015.

# Dependencies

* Python (>= 3.6)
* Numpy (>= 1.13.3)
* Cython (>= 0.26.1)
* scikit-learn (>= 0.19.1)

# Examples

(tbd.)

## Test

```
python setup.py build_ext --inplace
python test_path.py
```

# License

This software contains code from [scikit-learn](http://scikit-learn.org)
which is distrubuted under the 3-Clause BSD License:

> Copyright (c) 2007–2017 The scikit-learn developers.
>  All rights reserved.
>
>
> Redistribution and use in source and binary forms, with or without
> modification, are permitted provided that the following conditions are met:
>
>   a. Redistributions of source code must retain the above copyright notice,
>      this list of conditions and the following disclaimer.
>   b. Redistributions in binary form must reproduce the above copyright
>      notice, this list of conditions and the following disclaimer in the
>      documentation and/or other materials provided with the distribution.
>   c. Neither the name of the Scikit-learn Developers  nor the names of
>      its contributors may be used to endorse or promote products
>      derived from this software without specific prior written
>      permission.
>
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
> AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
> IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
> ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
> ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
> DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
> SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
> CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
> LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
> OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
> DAMAGE.
