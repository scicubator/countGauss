//static void treatQP(vector<float> &W, vector<float> &b, vector<float> &result);
//void set_values(dose_matrix, dose, weights);
using namespace std;

namespace Anchor
{
    class NQP
    {
        int n;
        int m;
        public:
            vector<double> W;
            NQP(vector<double> &, int &, int &);
            vector<double> anchor_fast(vector<double> &);
            void hello();

        private:
        //__global__ void elem_div(float *, float *, float *);
    };

}
