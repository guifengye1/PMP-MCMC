#include <iostream>
#include <random>
#include <vector>
#include <cuda_runtime.h>
#include <time.h>
#include <numeric>
#include <fstream>
#include <string>

__global__ void log_likelihood_kernel(float* x, float* y, float* gpu_a, float* nets, float* tran_table, int nets_num, int net_size, int data_num, int deep,int N_step) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const float MPI = 3.14159265359;
    //(N + 1) * tree_deep * N_step * 2
    if (idx < nets_num) {

        for (int i = 0; i < data_num; i++) {
            float y_hat = nets[idx * 3] + nets[idx * 3 + 1] * x[i];
            double temp = (y[i] - y_hat) / nets[idx * 3 + 2];
            gpu_a[idx] += (-0.5 * log(2 * MPI * nets[idx * 3 + 2] * nets[idx * 3 + 2]) - 0.5 * temp * temp)/2000.0;
        }

        for (int d = 0; d < deep; d++) {
            for (int step = 0; step < N_step; step++) {
                int net_from_index = tran_table[idx * deep * N_step * 2 + d * N_step * 2 + step * 2];
                int net_to_index = tran_table[idx * deep * N_step * 2 + d * N_step * 2 + step * 2 + 1];
                for (int j = 0; j < 3; j++) {
                    double temp = nets[net_from_index * 3 + j] - nets[net_to_index * 3 + j];
                    gpu_a[idx] += (-0.5 * log(2 * MPI) - 0.5 * temp * temp);
                }

            }
            
        }
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
    std::cout << "data size: X-" << X.size() << " Y-" << Y.size();
    data_x.close();
    data_y.close();

}


int main() {

    // 3,7,15,31,63,127,255,511,1023,2047,4095,8192
    int N = 511; //并行度设定
    int tree_deep = 3;  //树的深度
    int N_step = 7;
    
    
    int num_steps = 2000; //迭代次数设定
    float adjust_A = 60.0;
    // 3:  60

    int device = 3; // 要使用的设备编号，例如第一个GPU设备编号为0

    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(device);



    // 1.生成数据
   
    clock_t start, end;

    // 设置随机数生成器和分布
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> uniform_dist(-1.0f, 1.0f);
    std::normal_distribution<float> normal_dist(0.0f, 1.0f);


    // 生成随机数据
    std::vector<float> X;
    std::vector<float> Y;
    get_data(X, Y);



    // 2.开始迭代

    float alpha = 0.02; //迭代步长设定
    std::normal_distribution<float> updata(0.0f, alpha);  //更新函数
    std::vector<float> init_net = { 0,0,1 };  //初始参数设定
    std::vector<float> now_net = { 0,0,1 };   //记录当前参数
    std::vector<float> nets(now_net.size() * (N + 1), -1);  //每次迭代的待采参数
    std::vector<float> A(N + 1);   //采样权重
    std::vector<double> A_hat(N + 1);   //采样权重
    std::vector<int> sampled_values;  //根据权重获得的采样索引
    std::vector<float> data_log((N + 1) * (num_steps) * 3);   //参数记录
    std::vector<float> A_log((N + 1) * (num_steps));

    
    int tran_table_size = (N + 1) * tree_deep * N_step * 2;  //建立转移表，用于并行化转移部分计算
    std::vector<int> tran_table(tran_table_size, -1);
    float* gpu_x, * gpu_y, * gpu_a, * gpu_nets, * gpu_tran_table; //需要传给gpu的数据
    size_t bytes = X.size() * sizeof(float);
    size_t net_size = (N + 1) * sizeof(float);
    size_t nets_size = nets.size() * sizeof(nets[0]);
    size_t tran_size = tran_table_size * sizeof(float);
    // 在 GPU 上分配内存
    cudaMalloc(&gpu_x, bytes);
    cudaMalloc(&gpu_y, bytes);
    cudaMalloc(&gpu_a, net_size);
    cudaMalloc(&gpu_nets, nets_size);
    cudaMalloc(&gpu_tran_table, tran_size);
    // 将数据从主机内存复制到 GPU
    cudaMemcpy(gpu_x, X.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_y, Y.data(), bytes, cudaMemcpyHostToDevice);

    // 打开文件进行写操作
    // 将整数转换为字符串
    std::string strAlgo = "PMP_";
    std::string strNumSteps = std::to_string(num_steps);

    std::string strbeta0 = strAlgo + "beta0_" + strNumSteps + ".txt";
    std::string strbeta1 = strAlgo + "beta1_" + strNumSteps + ".txt";
    std::string strsigma = strAlgo + "sigma_" + strNumSteps + ".txt";

    std::string strtime = strAlgo + "time" + strNumSteps + ".txt";
    std::ofstream outbeta0(strbeta0);
    std::ofstream outbeta1(strbeta1);
    std::ofstream outsigma(strsigma);

    std::ofstream outstrtime(strtime);
    start = clock();  // 开始计时
    float time = 0.0;
    std::vector<int> from_set(N_step + 1);// 存储索引
    start = clock();  // 开始计时
    // 迭代开始
    for (int i = 0; i < num_steps; ++i) {
       
        // 1.产生建议参数
            // 1.1 初始化模型
        sampled_values.clear();
        std::fill(A.begin(), A.end(), 0.0);
        cudaMemcpy(gpu_a, A.data(), net_size, cudaMemcpyHostToDevice);

        for (int state = 0; state < now_net.size(); state++) {
            nets[state] = now_net[state];
        }
        // 1.2 生成建议模型和转移矩阵
        for (int deep = 0; deep < tree_deep; deep++) {
            int temp = pow((N_step + 1), deep);
            
            for (int k = 0; k < temp; k++) {
                
                from_set[0] = k;
                for (int j = 0; j < N_step; j++) {
                    int from_index = k;
                    int to_index = k + temp * (j + 1);
                    from_set[j + 1] = to_index;
                    for (int state = 0; state < now_net.size(); state++) {
                        nets[to_index*3 + state] = nets[from_index*3 + state] + updata(gen);
                    }
                   

                }
                //更新表
                for (int table_start_index = 0; table_start_index < from_set.size(); table_start_index++) { 
                    int end_index = 0;
                    for (int table_end_index = 0; table_end_index < from_set.size(); table_end_index++) {
                        if (from_set[table_start_index] != from_set[table_end_index]) {
                            
                            tran_table[from_set[table_start_index] * tree_deep * N_step * 2+ deep * N_step * 2 + end_index *2] = from_set[table_start_index];
                            tran_table[from_set[table_start_index] * tree_deep * N_step * 2 + deep * N_step * 2 + end_index * 2 + 1] = from_set[table_end_index];
                            end_index++;
                        }
                        
                    }
                    if (deep - 1 > -1 && tran_table[from_set[table_start_index] * tree_deep * N_step * 2 + (deep - 1)* N_step * 2] == -1) {
                        for (int index = 0; index < deep; index++) {//遍历深度
                            for (int table_end_index = 0; table_end_index < N_step; table_end_index++) {//遍历目标点
                                tran_table[from_set[table_start_index] * tree_deep * N_step * 2 + index * N_step * 2 + table_end_index * 2] = tran_table[from_set[0] * tree_deep * N_step * 2 + index * N_step * 2 + table_end_index * 2];
                                tran_table[from_set[table_start_index] * tree_deep * N_step * 2 + index * N_step * 2 + table_end_index * 2+1] = tran_table[from_set[0] * tree_deep * N_step * 2 + index * N_step * 2 + table_end_index * 2+1];
                            }
                        }
                    }
                }
                
            }
        }
 
            
        

        cudaMemcpy(gpu_nets, nets.data(), nets_size, cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_tran_table, tran_table.data(), tran_table_size, cudaMemcpyHostToDevice);

        // 2.优化采样
            // 2.1 计算接受率
        int nets_num = N + 1;
        int datas_num = X.size();
        // 定义 GPU 线程块大小和数量
        int blockSize = 256;
        int gridSize = (nets_num + blockSize - 1) / blockSize;

        log_likelihood_kernel << <gridSize, blockSize >> > (gpu_x, gpu_y, gpu_a, gpu_nets, gpu_tran_table, nets_num, net_size, datas_num, tree_deep, N_step);
        //接受率返回cpu  
        cudaMemcpy(A.data(), gpu_a, net_size, cudaMemcpyDeviceToHost);
        // 标准化数组 A
         //standardize(A);

        for (int c = 0; c < A.size(); c++) {
            A_hat[c] = exp(double(A[c]) + double(adjust_A));

        }
        


        double sum = std::accumulate(A_hat.begin(), A_hat.end(), 0);


        // 2.2 根据权重抽样
    // 创建离散分布，使用 B 中的权重
        std::discrete_distribution<> d(A_hat.begin(), A_hat.end());
        // 进行 N+1 次采样
        for (int i = 0; i < N + 1; ++i) {
            sampled_values.push_back(d(gen));
        }


        //记录参数
        
        for (int index = 0; index < (N + 1); index++) {
            // 更新参数

            A_log[(i) * (N + 1) + index] = A_hat[index] / sum;

            data_log[(i) * (N + 1) * 3 + index * 3] = nets[sampled_values[index] * 3];
            data_log[(i) * (N + 1) * 3 + index * 3 + 1] = nets[sampled_values[index] * 3 + 1];
            data_log[(i) * (N + 1) * 3 + index * 3 + 2] = nets[sampled_values[index] * 3 + 2];
        }

        
        
        // 更新参数
        for (int state = 0; state < now_net.size(); state++) {
            now_net[state] = nets[sampled_values[0] * 3 + state];
           

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



}
