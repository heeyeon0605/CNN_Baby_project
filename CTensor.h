#pragma once
#include "Imagelib.h"

class Tensor3D {
private:
	double*** tensor;
	int nH; // height
	int nW; // width
	int nC; // channel
public:
	Tensor3D(int _nH, int _nW, int _nC) : nH(_nH), nW(_nW), nC(_nC) {
		tensor = dmatrix3D(_nH, _nW, _nC);
	}
	~Tensor3D() {
		free_dmatrix3D(tensor, nH, nW, nC);
	}
	void set_elem(int _h, int _w, int _c, double _val) {
		tensor[_h][_w][_c] = _val;
	}

	double get_elem(int _h, int _w, int _c)	const {
		return tensor[_h][_w][_c];
	}

	void get_info(int& _nH, int& _nW, int& _nC) const {
		_nH = nH;
		_nW = nW;
		_nC = nC;
	}

	void set_tensor(double*** _tensor) { tensor = _tensor; }
	double*** get_tensor() const { return tensor; }

	void print() const {
		cout << nH << "*" << nW << "*" << nC << endl;
	}
};
