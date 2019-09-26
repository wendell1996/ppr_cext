#include "ppr.h"

template <class I,class T>
void dot_merge(const I n_row,
                     I Cp[],
                     I Cj[],
                     T Cx[]);
//template <class I>
//void _count_nnz_job(const I n_row,
//                    const I n_col,
//                    const I Ap[],
//                    const I Aj[],
//                    const I Bp[],
//                    const I Bj[],
//                          I Cp[],
//                    const I idx_start,
//                    const I idx_end);

void template_instance(){
    long long k;
    int ki;
    long long Lp[1];
    int Ip[1];
    double Dx[1];
    float Fx[1];
    top_k_dot(ki,ki,ki,Ip,Ip,Dx,Ip,Ip,Dx,Ip,Ip,Dx);
    top_k_dot(k,k,k,Lp,Lp,Dx,Lp,Lp,Dx,Lp,Lp,Dx);
    top_k_dot(ki,ki,ki,Ip,Ip,Fx,Ip,Ip,Fx,Ip,Ip,Fx);
    top_k_dot(k,k,k,Lp,Lp,Fx,Lp,Lp,Fx,Lp,Lp,Fx);

    count_nnz(ki,ki,Ip,Ip,Ip,Ip,Ip);
    
//    _count_nnz_job(ki,ki,Ip,Ip,Ip,Ip,Ip,ki,ki);
    count_nnz_parallel(ki,ki,Ip,Ip,Ip,Ip,Ip,1);
    auto dk_i_f = dot_parallel<int,float>;
    auto sq_i_f = squeeze_k_parallel<int,float>;
}

template <class I>
void count_nnz(const I n_row,
               const I n_col,
               const I Ap[],
               const I Aj[],
               const I Bp[],
               const I Bj[],
                     I Cp[])
{
    std::vector<I> count(n_col,-1);
    I nnz{0};
    Cp[0] = 0;
    for(I idx_row = 0; idx_row < n_row; ++idx_row){
        const I& i_a_start = Ap[idx_row];
        const I& i_a_end   = Ap[idx_row+1];
        I row_nnz{0};
        for(I i_a_idx = i_a_start; i_a_idx < i_a_end; ++i_a_idx){
            const I& i_j_a     = Aj[i_a_idx];
            const I& j_b_start = Bp[i_j_a];
            const I& j_b_end   = Bp[i_j_a+1];
            for(I j_b_idx = j_b_start; j_b_idx < j_b_end; ++j_b_idx){
                const I& j_k_b = Bj[j_b_idx];
                if(count[j_k_b] != idx_row){
                    count[j_k_b] = idx_row;
                    ++row_nnz;
                }
            }
        }
        nnz += row_nnz;
        Cp[idx_row+1] = nnz;
    }   
}

template <class I>
void _count_nnz_job(const I n_row,
                    const I n_col,
                    const I Ap[],
                    const I Aj[],
                    const I Bp[],
                    const I Bj[],
                          I Cp[],
                    const I idx_start,
                    const I idx_end)
{ 
    std::vector<I> count(n_col,-1);
    for(I idx_row = idx_start; idx_row < idx_end; ++idx_row){
        const I& i_a_start = Ap[idx_row];
        const I& i_a_end   = Ap[idx_row+1];
        I row_nnz{0};
        for(I i_a_idx = i_a_start; i_a_idx < i_a_end; ++i_a_idx){
            const I& i_j_a     = Aj[i_a_idx];
            const I& j_b_start = Bp[i_j_a];
            const I& j_b_end   = Bp[i_j_a+1];
            for(I j_b_idx = j_b_start; j_b_idx < j_b_end; ++j_b_idx){
                const I& j_k_b = Bj[j_b_idx];
                if(count[j_k_b] != idx_row){
                    count[j_k_b] = idx_row;
                    ++row_nnz;
                }
            }
        }
        Cp[idx_row+1] = row_nnz;
    }
}

template <class I>
void count_nnz_merge(const I n_row,
                           I Cp[])
{
    for(I i = 1; i < n_row+1; ++i) Cp[i]+=Cp[i-1];
}

template <class I>
void count_nnz_parallel(const I   n_row,
                        const I   n_col,
                        const I   Ap[],
                        const I   Aj[],
                        const I   Bp[],
                        const I   Bj[],
                              I   Cp[],
                        const int num_jobs)
{
    if(num_jobs <= 0 || n_row <= num_jobs){
        count_nnz(n_row,
                  n_col,
                  Ap,
                  Aj,
                  Bp,
                  Bj,
                  Cp);
    }else{
        std::vector<std::thread> threads;
        Cp[0] = 0;
        double d_num_jobs  = static_cast<double>(num_jobs);
        double d_n_row     = static_cast<double>(n_row);
        for(I i = 0; i < num_jobs; ++i){
            double d_i  = static_cast<double>(i);
            I idx_start = static_cast<I>(d_n_row*d_i/d_num_jobs);
            I idx_end   = static_cast<I>(d_n_row*(d_i+1)/d_num_jobs);
            std::thread t(_count_nnz_job<I>,
                          n_row,
                          n_col,
                          Ap,
                          Aj,
                          Bp,
                          Bj,
                          Cp,
                          idx_start,
                          idx_end);
            threads.push_back(std::move(t));
        }
        for(auto& t:threads) t.join();
        count_nnz_merge(n_row, Cp);
    }
}

template <class I,class T>
void _dot_job(const I n_row,
              const I n_col,
              const I Ap[],
              const I Aj[],
              const T Ax[],
              const I Bp[],
              const I Bj[],
              const T Bx[],
                    I Cp[],
                    I Cj[],
                    T Cx[],
              const I idx_start,
              const I idx_end)
{
    std::vector<T> sums(n_col,0);
    std::vector<I> next(n_col,-1);
    I nnz{0};
    //clock_t t1,t2,t3;
    for(I idx_row = idx_start; idx_row < idx_end; ++idx_row){
        I  nnz   = Cp[idx_row];
        I head   = -2;
        I length = 0;
        const I& i_a_start = Ap[idx_row];
        const I& i_a_end   = Ap[idx_row+1];
        //std::unordered_set<I> indices;
        for(I i_a_idx = i_a_start; i_a_idx < i_a_end; ++i_a_idx){
            const I& i_j_a      = Aj[i_a_idx];
            const I& j_b_start  = Bp[i_j_a];
            const I& j_b_end    = Bp[i_j_a+1];
            const T& i_j_a_data = Ax[i_a_idx];
            for(I j_b_idx = j_b_start; j_b_idx < j_b_end; ++j_b_idx){
                const I& j_k_b      = Bj[j_b_idx];
                const T& j_k_b_data = Bx[j_b_idx];
                //t1 = clock();
                sums[j_k_b] += i_j_a_data * j_k_b_data;
                if(next[j_k_b] == -1){
                    next[j_k_b] = head;
                    head = j_k_b;
                    ++length;
                }
                //t2 = clock();
               // if(indices.count(j_k_b)==0){
               //     indices.insert(j_k_b);
               // }
                //t3 = clock();
            }
        }
        
        for(I i = 0; i< length; ++i){
            if(fabs(sums[head]) >= 1e-16){
                Cj[nnz] = head;
                Cx[nnz++] = sums[head];
            }else{
                Cj[nnz++] = -1;
            }

            I temp = head;
            head = next[head];
            next[temp] = -1;
            sums[temp] = 0;
        }
       // for(const auto& idx:indices){
       //     if(sums[idx] == 0){
       //         Cj[nnz++] = -1;
       //         continue;
       //     }
       //     Cj[nnz] = idx;
       //     Cx[nnz++] = sums[idx];
       //     sums[idx] = 0;
       // }
    }
    //std::cerr << "1: " << t2-t1 << "2: " << t3-t2 << std::endl;
}

template <class I,class T>
void squeeze_k_job(const I n_row,
                   const I k,
                         I Cp[],
                         I Cj[],
                         T Cx[],
                   const I idx_start,
                   const I idx_end)
{
    using top_k_pair = std::pair<T,I>;
    using top_k_queue =
        std::priority_queue<top_k_pair,std::vector<top_k_pair>,std::greater<top_k_pair>>;

    for(I idx_row = idx_start; idx_row < idx_end; ++idx_row){
        const I& cur_start = Cp[idx_row];
        const I& cur_end   = Cp[idx_row+1];
        if(cur_end - cur_start <= k) continue;
        top_k_queue top_k;
        I size{0};
        for(I idx_col = cur_start; idx_col < cur_end; ++idx_col){
            if(Cj[idx_col] >= 0){
                if(size < k){
                    ++size;
                    top_k.emplace(std::make_pair(Cx[idx_col],idx_col));
                }else if(top_k.top().first < Cx[idx_col]){
                    top_k.emplace(std::make_pair(Cx[idx_col],idx_col));
                    Cj[top_k.top().second] = -1;
                    top_k.pop();
                }else{
                    Cj[idx_col] = -1;
                }
            }
        }
    } 
}

template <class I, class T>
void squeeze_k_parallel(const I   n_row,
                        const I   k,
                              I   Cp[],
                              I   Cj[],
                              T   Cx[],
                        const int num_jobs)
{
    if(num_jobs < 0 || num_jobs >= n_row){
        squeeze_k_job(n_row,
                      k,
                      Cp,
                      Cj,
                      Cx,
                      0,
                      n_row);
    }else{
        std::vector<std::thread> threads;
        Cp[0] = 0;
        double d_num_jobs  = static_cast<double>(num_jobs);
        double d_n_row     = static_cast<double>(n_row);
        for(I i = 0; i < num_jobs; ++i){
            double d_i  = static_cast<double>(i);
            I idx_start = static_cast<I>(d_n_row*d_i/d_num_jobs);
            I idx_end   = static_cast<I>(d_n_row*(d_i+1)/d_num_jobs);
            std::thread t(squeeze_k_job<I,T>,
                          n_row,
                          k,
                          Cp,
                          Cj,
                          Cx,
                          idx_start,
                          idx_end);
            threads.push_back(std::move(t));
        }
        for(auto& t:threads) t.join();
    }
    dot_merge(n_row,
              Cp,
              Cj,
              Cx);
}

template <class I,class T>
void dot_merge(const I n_row,
                     I Cp[],
                     I Cj[],
                     T Cx[])
{
    I nnz = 0;
    Cp[0] = 0;
    I cur_start = 0;
    I cur_end;
    for(I idx_row = 0; idx_row < n_row; ++idx_row){
        cur_end = Cp[idx_row+1];
        for(I idx_col = cur_start; idx_col < cur_end; ++idx_col){
            if(Cj[idx_col] >= 0){
                if(nnz == idx_col){
                    ++nnz;
                }else{
                    Cj[nnz]   = Cj[idx_col];
                    Cx[nnz++] = Cx[idx_col];
                }
            }
        }
        cur_start = cur_end;
        Cp[idx_row+1] = nnz;
    }
}

template <class I,class T>
void dot_parallel(const I   n_row,
                    const I   n_col,
                    const I   Ap[],
                    const I   Aj[],
                    const T   Ax[],
                    const I   Bp[],
                    const I   Bj[],
                    const T   Bx[],
                          I   Cp[],
                          I   Cj[],
                          T   Cx[],
                    const int num_jobs)
{
    if(num_jobs <= 0 || n_row <= num_jobs){
        _dot_job(n_row,
                 n_col,
                 Ap,
                 Aj,
                 Ax,
                 Bp,
                 Bj,
                 Bx,
                 Cp,
                 Cj,
                 Cx,
                 0,
                 n_row);
    }else{
        std::vector<std::thread> threads;
        Cp[0] = 0;
        double d_num_jobs  = static_cast<double>(num_jobs);
        double d_n_row     = static_cast<double>(n_row);
        for(I i = 0; i < num_jobs; ++i){
            double d_i  = static_cast<double>(i);
            I idx_start = static_cast<I>(d_n_row*d_i/d_num_jobs);
            I idx_end   = static_cast<I>(d_n_row*(d_i+1)/d_num_jobs);
            std::thread t(_dot_job<I,T>,
                          n_row,
                          n_col,
                          Ap,
                          Aj,
                          Ax,
                          Bp,
                          Bj,
                          Bx,
                          Cp,
                          Cj,
                          Cx,
                          idx_start,
                          idx_end);
            threads.push_back(std::move(t));
        }
        for(auto& t:threads) t.join();
    }
    dot_merge(n_row,
              Cp,
              Cj,
              Cx);
}

template <class I,class T>
void top_k_dot(const I n_row,
               const I n_col,
               const I k,
               const I Ap[],
               const I Aj[],
               const T Ax[],
               const I Bp[],
               const I Bj[],
               const T Bx[],
                     I Cp[],
                     I Cj[],
                     T Cx[])
{
    using top_k_pair = std::pair<T,npy_intp>;
    using top_k_queue =
        std::priority_queue<top_k_pair,std::vector<top_k_pair>,std::greater<top_k_pair>>;
    I nnz{0};
    Cp[0] = 0;
    const I i_a_start = 0;
    const I i_a_end   = n_row;

    for(I i = i_a_start; i < i_a_end; ++i){
        std::unordered_set<I> indices;
        //std::unordered_set<I>().swap(indices);
        //std::vector<T> sums(n_col,0);
        T sums[n_col];
        for(int n=0;n<n_col;++n) sums[n]=0;

        const I& i_col_a_start = Ap[i];
        const I& i_col_a_end   = Ap[i+1];
        for(I i_a_idx = i_col_a_start; i_a_idx < i_col_a_end; ++i_a_idx){
            const I& i_j_a      = Aj[i_a_idx];
            const T& i_j_a_data = Ax[i_a_idx];

            const I& j_col_b_start = Bp[i_j_a];
            const I& j_col_b_end   = Bp[i_j_a+1];
            for(I j_b_idx = j_col_b_start; j_b_idx < j_col_b_end;
                    ++j_b_idx){
                const I& j_k_b      = Bj[j_b_idx];
                const T& j_k_b_data = Bx[j_b_idx];
                sums[j_k_b] += i_j_a_data * j_k_b_data;
                if(indices.count(j_k_b) == 0){
                    indices.insert(j_k_b);
                }
            }
        }
        top_k_queue top_k;
        npy_intp row_nnz = 0;
        for(const auto& idx:indices){
            if(row_nnz++ < k){
                top_k.emplace(std::make_pair(sums[idx],idx));
            }else if(top_k.top().first < sums[idx]){
                top_k.emplace(std::make_pair(sums[idx],idx));
                top_k.pop();
            }
        }
        while(!top_k.empty()){
            auto  data_idx = top_k.top();
            top_k.pop();
            T& data        = data_idx.first;
            auto& idx      = data_idx.second;
            if(fabs(data) < 1e-16){
                continue;
            }else{
                Cj[nnz]   = idx;
                Cx[nnz++] = data;
            }
        }
        Cp[i+1] = nnz;
    }
}
