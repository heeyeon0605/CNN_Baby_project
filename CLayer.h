#pragma once
#include "Imagelib.h"
#include "CTensor.h"

#define MEAN_INIT 0
#define LOAD_INIT 1


class Layer {
protected:
	int fK; // kernel size in K*K kernel
	int fC_in; // number of channels
	int fC_out; //number of filters
	string name;
public:
	Layer(string _name, int _fK, int _fC_in, int _fC_out) : name(_name), fK(_fK), fC_in(_fC_in), fC_out(_fC_out) {}
	virtual ~Layer() {};
	virtual Tensor3D* forward(const Tensor3D* input) = 0;
	//	virtual bool backward() = 0;
	virtual void print() const = 0;
	virtual void get_info(string& _name, int& _fK, int& _fC_in, int& _fC_out) const = 0;
};


class Layer_ReLU : public Layer {
public:
	Layer_ReLU(string _name, int _fK, int _fC_in, int _fC_out) :Layer(_name, _fK, _fC_in, _fC_out)
	{}
	~Layer_ReLU() {}

	Tensor3D* forward(const Tensor3D* input) override {
		int nH; // height
		int nW; // width
		int nC; // channel

		double x;

		input->get_info(nH, nW, nC);
		Tensor3D* output = new Tensor3D(nH, nW, nC);

		for (int h = 0; h < nH; h++) {
			for (int w = 0; w < nW; w++) {
				for (int c = 0; c < nC; c++) {
					x = input->get_elem(h, w, c);
					if (x < 0) x = 0;

					output->set_elem(h, w, c, x);
				}
			}
		}

		cout << name << " is finished" << endl;
		return output;
	};
	void get_info(string& _name, int& _fK, int& _fC_in, int& _fC_out) const override {
		_fK = fK;
		_fC_in = _fC_in;
		_fC_out = _fC_out;
	}
	void print() const override {
		cout << "Relu1:" << '\t' << fK << "*" << fK << "*" << fC_in << "*" << fC_out << endl;
	}
};


class Layer_Conv : public Layer {
private:
	string filename_weight;
	string filename_bias;
	double**** weight_tensor;
	double* bias_tensor;
public:
	Layer_Conv(string _name, int _fK, int _fC_in, int _fC_out, int init_type, string _filename_weight = "", string _filename_bias = "") :Layer(_name, _fK, _fC_in, _fC_out), filename_weight(_filename_weight), filename_bias(_filename_bias) {
		init(init_type);
	}
	void init(int init_type) {
		double x;
		double y;
		string s;
		string s_2;

		weight_tensor = dmatrix4D(fK, fK, fC_in, fC_out);
		bias_tensor = dmatrix1D(fC_out);

		if (init_type == MEAN_INIT) {
			int offset = (fK - 1) / 2;
			x = 1.0 / (fK * fK * fC_in);

			for (int o = 0; o < fC_out; o++) {
				for (int c = 0; c < fC_in; c++) {
					for (int ph = 0; ph < fK; ph++) {
						for (int pw = 0; pw < fK; pw++) {
							weight_tensor[ph][pw][c][o] = x;
						}
					}
				}
			}

			for (int f = 0; f < fC_out; f++)
				bias_tensor[f] = 0;
		}
		
		if (init_type == LOAD_INIT) {
			ifstream fin;
			fin.open(filename_weight);

			if (!fin) {
				cout << "file not find : " << filename_weight;
				exit(100);
			}

			for (int o = 0; o < fC_out; o++) {
				for (int c = 0; c < fC_in; c++) {
					for (int ph = 0; ph < fK; ph++) {
						for (int pw = 0; pw < fK; pw++) {
							getline(fin, s);
							x = atof(s.c_str());
							weight_tensor[ph][pw][c][o] = x;
						}
					}
				}
			}
			
			fin.close();
			
			fin.open(filename_bias);

			if (!fin) {
				cout << "file not find : " << filename_bias;
				exit(100);
			}

			for (int f = 0; f < fC_out; f++) {
				getline(fin, s_2);
				y = atof(s_2.c_str());
				bias_tensor[f] = y;
			}
			fin.close();
		}

	}
	~Layer_Conv() override {
		free_dmatrix4D(weight_tensor, fK, fK, fC_in, fC_out);
		free_dmatrix1D(bias_tensor, fC_out);
	}
	Tensor3D* forward(const Tensor3D* input) override {
		int nH; // height
		int nW; // width
		int nC; // channel

		double x;


		input->get_info(nH, nW, nC);
		Tensor3D* output = new Tensor3D(nH, nW, fC_out);

		int offset = (fK - 1) / 2;

		for (int h = offset; h < nH - offset; h++) {
			for (int w = offset; w < nW - offset; w++) {
					for (int o = 0; o < fC_out; o++) {
						for (int ph = 0; ph < fK; ph++) {
							for (int pw = 0; pw < fK; pw++) {
								for (int c = 0; c < nC; c++) {
								x = output->get_elem(h, w, o);
								x += input->get_elem(h + ph - offset, w + pw - offset, c) * weight_tensor[ph][pw][c][o];
								output->set_elem(h, w, o, x);
							}
						}
						output->set_elem(h, w, o, output->get_elem(h, w, o) + bias_tensor[o]);
					}
				}
			}
		}

		cout << name << " is finished" << endl;
		return output;
	};
	void get_info(string& _name, int& _fK, int& _fC_in, int& _fC_out) const override {
		_fK = fK;
		_fC_in = _fC_in;
		_fC_out = _fC_out;
	}
	void print() const override {
		cout << "Conv1:" << '\t' << fK << "*" << fK << "*" << fC_in << "*" << fC_out << endl;
	}
};
