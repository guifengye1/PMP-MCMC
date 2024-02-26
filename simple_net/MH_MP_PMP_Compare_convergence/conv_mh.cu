#include <iostream>
#include <random>
#include <vector>
#include <cuda_runtime.h>
#include <time.h>
#include <numeric>
#include <fstream>
#include <string>

__global__ void log_likelihood_kernel(float *x,float *y, float* gpu_a,float * gpu_nets,int data_num) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const float MPI = 3.14159265359;
    
   
        
    for (int i = 0; i < data_num; i++) {
        float y_hat = gpu_nets[0] + gpu_nets[1] * x[i];
        double temp = (y[i] - y_hat) / gpu_nets[2];
        gpu_a[0] += (-0.5 * log(2 * MPI * gpu_nets[2] * gpu_nets[2]) - 0.5 * temp * temp)/2000;      
    }
    for (int i = 0; i < data_num; i++) {
        float y_hat = gpu_nets[3] + gpu_nets[4] * x[i];
        double temp = (y[i] - y_hat) / gpu_nets[5];
        gpu_a[1] += (-0.5 * log(2 * MPI * gpu_nets[5] * gpu_nets[5]) - 0.5 * temp * temp)/2000;
    }    
}

// 函数用于计算均值
float calculateMean(const std::vector<float>& data) {
    float sum = std::accumulate(data.begin(), data.end(), 0.0);
    return sum / data.size();
}

// 函数用于计算标准差
double calculateStdDev(const std::vector<float>& data, float mean) {
    double sq_sum = std::inner_product(data.begin(), data.end(), data.begin(), 0.0);
    double variance = sq_sum / data.size() - mean * mean;
    // 如果方差小于零，则将其设为零
    if (variance < 0) {
        variance = 0;
    }
    return std::sqrt(variance);
}
// 函数用于标准化数组
void standardize(std::vector<float>& A) {
    float mean = calculateMean(A);
    double std_dev = calculateStdDev(A, mean);
    for (float& val : A) {        
        val = exp((val - mean) / std_dev);
    }
}
void get_data(std::vector<float>& X, std::vector<float>& Y) {
    // 读取数据
    float num_x;
    float num_y;
    std::ifstream data_x("data_x.txt");
    std::ifstream data_y("data_y.txt");
 
    while (data_x >> num_x) {
        X.push_back(num_x);
    }
    while (data_y >> num_y) {
        Y.push_back(num_y);
    }
    std::cout << "data size: X-" << X.size() <<" Y-" << Y.size();
    data_x.close();
    data_y.close();

}


int main() {
    int num_steps = 2000;//迭代次数设定
    clock_t start, end;
    int device = 0; // 要使用的设备编号，例如第一个GPU设备编号为0

    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(device);


    // 设置随机数生成器和分布
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> uniform_dist(-1.0f, 1.0f);
    std::normal_distribution<float> normal_dist(0.0f, 1.0f);

    // 生成随机数据
    std::vector<float> X;
    std::vector<float> Y;
    get_data(X, Y);

    // 打开文件进行写操作
    // 将整数转换为字符串
    std::string strAlgo = "MH_";
    std::string strNumSteps = std::to_string(num_steps);
   
    std::string strbeta0 = strAlgo + "beta0_"+ strNumSteps + ".txt";
    std::string strbeta1 = strAlgo + "beta1_" + strNumSteps + ".txt";
    std::string strsigma = strAlgo + "sigma_" + strNumSteps + ".txt";

    std::string strtime = strAlgo + "time" + strNumSteps + ".txt";
    std::ofstream outbeta0(strbeta0);
    std::ofstream outbeta1(strbeta1);
    std::ofstream outsigma(strsigma);

    std::ofstream outstrtime(strtime);
    float time = 0.0;

    // 2.开始迭代
    
    float alpha = 0.02; //迭代步长设定
    std::normal_distribution<float> updata(0.0f, alpha);  //更新函数
    std::vector<float> new_net = { 0,0,1,0,0,1 };  //初始参数设定
    std::vector<float> now_net = { 0,0,1 };   //记录当前参数
    std::vector<float> time_log(num_steps );  // 记录时间
    std::vector<float> data_log(num_steps * 3);   //参数记录
    std::vector<float> A(2);   //采样权重

    float* gpu_x, * gpu_y, * gpu_nets, * gpu_a; //需要传给gpu的数据
    size_t bytes = X.size() * sizeof(float);
    size_t net_size = 2 * sizeof(float);
    size_t nets_size = new_net.size() * sizeof(new_net[0]);
    // 在 GPU 上分配内存
    cudaMalloc(&gpu_x, bytes);
    cudaMalloc(&gpu_y, bytes);
    cudaMalloc(&gpu_nets, nets_size);
    cudaMalloc(&gpu_a, net_size);
    // 将数据从主机内存复制到 GPU
    cudaMemcpy(gpu_x, X.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_y, Y.data(), bytes, cudaMemcpyHostToDevice);
    int datas_num = X.size();
    start = clock();  // 开始计时
    // 迭代开始
    for (int i = 0; i < num_steps; ++i) {
        std::fill(A.begin(), A.end(), 0.0);
        cudaMemcpy(gpu_a, A.data(), net_size, cudaMemcpyHostToDevice);

        // 产生新的建议值
        for (int state = 0; state < now_net.size(); state++) {
            new_net[state] = now_net[state];
            new_net[now_net.size() + state] = now_net[state] + updata(gen);
        }
        cudaMemcpy(gpu_nets, new_net.data(), nets_size, cudaMemcpyHostToDevice);
        int blockSize = 1;
        int gridSize = 1;

        log_likelihood_kernel << <gridSize, blockSize >> > (gpu_x, gpu_y, gpu_a, gpu_nets, datas_num);
        //接受率返回cpu  
        cudaMemcpy(A.data(), gpu_a, net_size, cudaMemcpyDeviceToHost);
        // 接受或者拒绝
        float random = rand() % (10000 + 1) / (float)(10000 + 1);
        if (random < exp(A[1] - A[0])) {
            for (int state = 0; state < now_net.size(); state++) {
                now_net[state] = new_net[now_net.size() + state];
            }
        }
        //运行结束计时
        end = clock();
        time = double(end - start) / CLOCKS_PER_SEC;

        outstrtime << time << "\n";
        outbeta0 << now_net[0] << "\n";
        outbeta1 << now_net[1] << "\n";
        outsigma << now_net[2] << "\n";

        
    }
    outbeta0.close();
    outbeta1.close();
    outsigma.close();
    outstrtime.close();
    return 1;
    

}
