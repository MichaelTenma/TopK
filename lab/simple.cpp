#include <mpi.h>
#include <CL/sycl.hpp>
#include<bits/stdc++.h>

#include <sys/time.h>　

using namespace cl::sycl;
using namespace std;

sycl::queue q;
const long long N = 100000000;


#define NUM int
#define MPI_NUM MPI_INT


const int K = 144;
random_device rd;
priority_queue<int> tq;
uniform_int_distribution<long long> dist(0, 2e9);
mt19937 gen(rd());

long long randx() {
    return dist(gen);
}

vector<int> now;

clock_t start, en;

int main(int argc, char *argv[]) {
    puts("-----------------------------------------------------------------");
    cout<<"N值为："<<N<<' '<<"K值为："<<K<<'\n';
    puts("-----------------------------------------------------------------");
    int *a = malloc_shared<int>(N, q);//存放数据
    int *b = malloc_shared<int>(N, q);//分块起始点存放
    int *c = malloc_shared<int>(N, q);//分块终点存放
    int *d = malloc_shared<int>(N, q);//分块K值存放
    for (long long i = 0; i < N; i++) {
        a[i] = randx();
        now.push_back(a[i]);
    }
    struct timeval t1, t2;


    gettimeofday(&t1, NULL);
    int st = 1;
    b[st] = 0;
    c[st] = now.size() - 1;
    d[st] = K;
    while (st) {
        int begin = b[st];
        int end = c[st];
        int k = d[st];
        st--;
        if (begin < end) {
            int baseNum = now[begin];
            int i = begin;
            int j = end;
            while (i < j) {
                while (i < j && now[j] >= baseNum) {
                    j--;
                }
                now[i] = now[j];
                while (i < j && now[i] <= baseNum) {
                    i++;
                }
                now[j] = now[i];
            }
            now[i] = baseNum;
            if (i - begin + 1 == k) {}
            else if (i - begin + 1 > k) {
                st++;
                b[st] = begin;
                c[st] = i - 1;
                d[st] = k;
            } else {
                st++;
                b[st] = i + 1;
                c[st] = end;
                d[st] = k - (i - begin + 1);
            }
        }
    }
    gettimeofday(&t2, NULL);
    long long timeuse =
            1000000 * ((long long) t2.tv_sec - (long long) t1.tv_sec) + (long long) t2.tv_usec - (long long) t1.tv_usec;
    cout << "串行快速选择—topk算法运行时间为： " << (double) timeuse / 1000000 << '\n';
    sort(now.begin(), now.begin() + K);
    reverse(now.begin(), now.begin() + K);
    for (int i = 0; i < K; i++) {
        cout << now[i] << ' ';
    }
    now.clear();
    cout << '\n';
    puts("-----------------------------------------------------------------");

    gettimeofday(&t1, NULL);
    for (int i = 0; i < N; i++) {
        tq.push(a[i]);
        if (tq.size() > K)tq.pop();
    }

    gettimeofday(&t2, NULL);
    timeuse =
            1000000 * ((long long) t2.tv_sec - (long long) t1.tv_sec) + (long long) t2.tv_usec - (long long) t1.tv_usec;
    cout << "串行堆排序—topk算法运行时间为： " << (double) timeuse / 1000000 << '\n';
    while (tq.size()) {
        int t = tq.top();
        tq.pop();
        cout << t << ' ';
    }
    cout << '\n';
    puts("-----------------------------------------------------------------");
    int B = sqrt(1ll * N * K);//每块大小B
    int L = N / B;//分成L块
    gettimeofday(&t1, NULL);
    q.parallel_for(range<1>(L), [=](id<1> u) {//对L个块并行

        int st = u * B + 1;//并行内核，不支持递归，因此采用栈实现，st表示栈头
        b[st] = u * B;// 每块的起始点
        c[st] = u * B + B - 1;//每块的终点
        d[st] = K;//每块所需的第K大
        while (st >= u * B + 1) {//判断栈是否为空
            int begin = b[st];
            int end = c[st];
            int k = d[st];
            st--;
            if (begin < end) {//快速选择算法
                int baseNum = a[begin];
                int i = begin;
                int j = end;
                while (i < j) {//后面小于基准点的移到前面来
                    while (i < j && a[j] >= baseNum) {
                        j--;
                    }
                    a[i] = a[j];
                    while (i < j && a[i] <= baseNum) {//前面大于基准点的移到后面去
                        i++;
                    }
                    a[j] = a[i];
                }//放置基准点
                a[i] = baseNum;
                if (i - begin + 1 == k) {}//刚好前K小找到
                else if (i - begin + 1 > k) {//找多了，继续找
                    st++;
                    b[st] = begin;
                    c[st] = i - 1;
                    d[st] = k;
                } else {//找少了，需要往后再找差值部分
                    st++;
                    b[st] = i + 1;
                    c[st] = end;
                    d[st] = k - (i - begin + 1);
                }
            }
        }
    }).wait();
    
    for (int i = 0; i < L; i++) {
        int f = i * B;
        for (int j = f; j < f + K; j++) {
            now.push_back(a[j]);
        }

    }

    st = 1;
    b[st] = 0;
    c[st] = now.size() - 1;
    d[st] = K;
    while (st) {
        int begin = b[st];
        int end = c[st];
        int k = d[st];
        st--;
        if (begin < end) {
            int baseNum = now[begin];
            int i = begin;
            int j = end;
            while (i < j) {
                while (i < j && now[j] >= baseNum) {
                    j--;
                }
                now[i] = now[j];
                while (i < j && now[i] <= baseNum) {
                    i++;
                }
                now[j] = now[i];
            }
            now[i] = baseNum;
            if (i - begin + 1 == k) {}
            else if (i - begin + 1 > k) {
                st++;
                b[st] = begin;
                c[st] = i - 1;
                d[st] = k;
            } else {
                st++;
                b[st] = i + 1;
                c[st] = end;
                d[st] = k - (i - begin + 1);
            }
        }
    }
    gettimeofday(&t2, NULL);
    timeuse =
            1000000 * ((long long) t2.tv_sec - (long long) t1.tv_sec) + (long long) t2.tv_usec - (long long) t1.tv_usec;
    cout << "并行快速选择—topk算法运行时间为： " << (double) timeuse / 1000000 << '\n';
    sort(now.begin(), now.begin() + K);
    reverse(now.begin(), now.begin() + K);
    for (int i = 0; i < K; i++) {
        cout << now[i] << ' ';
    }
    return 0;
}
