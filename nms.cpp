#include <iostream>
#include <vector>



struct Object
{
    int x_min;
    int y_min;
    int x_max;
    int y_max;
    int label;
    float prob;
};

static void qsort_descent_inplace(std::vector<Object>& objects, int left, int right)
{
    int i = left;
    int j = right;
    float p = objects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (objects[i].prob > p)
            i++;

        while (objects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(objects[i], objects[j]);

            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(objects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(objects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}



float IOU(const Object& a, const Object& b)
{
    int areaA = (a.x_max - a.x_min) * (a.y_max - a.y_min);
    if (areaA<=0) return 0.;
    int areaB = (b.x_max - b.x_min) * (b.y_max - b.y_min);
    if (areaB<=0) return 0.;
    
    int intersectionMinX = std::max(a.x_min, b.x_min);
    int intersectionMinY = std::max(a.y_min, b.y_min);
    int intersectionMaxX = std::min(a.x_max, b.x_max);
    int intersectionMaxY = std::min(a.y_max, b.y_max);
    
    int intersectionArea = std::max(intersectionMaxY - intersectionMinY, 0) * \
                                std::max(intersectionMaxX - intersectionMinX, 0);
    
    
    return (float)intersectionArea / (areaA + areaB - intersectionArea);
}


/*
 nonMaxSuppression
- Parameters:
  - objects: an vector of bounding boxes and their scores
  - limit: the maximum number of boxes that will be selected
  - threshold: used to decide whether boxes overlap too much
 */
void nonMaxSuppression(std::vector<Object>& objects, int limit, float threshold, std::vector<int>& selected)
{
    selected.clear();
    qsort_descent_inplace(objects);
    
    for(int i=0; i<objects.size(); ++i)
    {
        const Object& a = objects[i];
        bool keep = true;
        for (size_t j=0; j<selected.size(); ++j) {
            const Object& b = objects[selected[j]];
            if (b.label != a.label) continue;//多目标
            if (IOU(a, b) > threshold)
                keep = false;
        }
        if(keep) selected.push_back(i);
    }
}



void make_dummy_input(std::vector<Object> & objects)
{
    Object obj1;
    Object obj2;
    Object obj3;
    Object obj4;
    Object obj5;
    Object obj6;
    
    obj1.x_min = 12;
    obj1.y_min = 12;
    obj1.x_max = 50;
    obj1.y_max = 55;
    obj1.label = 1;
    obj1.prob = 0.5;
    objects.push_back(obj1);
    
    obj2.x_min = 10;
    obj2.y_min = 11;
    obj2.x_max = 56;
    obj2.y_max = 52;
    obj2.label = 1;
    obj2.prob = 0.7;
    objects.push_back(obj2);
    
    obj3.x_min = 10;
    obj3.y_min = 11;
    obj3.x_max = 56;
    obj3.y_max = 52;
    obj3.label = 2;
    obj3.prob = 0.8;
    objects.push_back(obj3);
    
    obj4.x_min = 8;
    obj4.y_min = 13;
    obj4.x_max = 50;
    obj4.y_max = 49;
    obj4.label = 2;
    obj4.prob = 0.34;
    objects.push_back(obj4);
    
    obj5.x_min = 80;
    obj5.y_min = 44;
    obj5.x_max = 100;
    obj5.y_max = 222;
    obj5.label = 3;
    obj5.prob = 0.44;
    objects.push_back(obj5);
    
    obj6.x_min = 80;
    obj6.y_min = 44;
    obj6.x_max = 100;
    obj6.y_max = 222;
    obj6.label = 3;
    obj6.prob = 0.47;
    objects.push_back(obj6);
}



int main()
{
    
    std::vector<Object> objects;
    make_dummy_input(objects);
    std::vector<int> selected;
    
    
    nonMaxSuppression(objects, 100, 0.45, selected);
    
    int count = selected.size();
    
    std::cout <<"num of objs: " << count << std::endl;
    std::vector<Object> selected_objects;
    selected_objects.resize(count);
    
    for(int i=0; i<count; ++i){
        selected_objects[i] = objects[selected[i]];
        std::cout << selected_objects[i].prob << ", " <<  selected_objects[i].label<< std::endl;
    }
        
    
    
    
    
    
    
    
    std::cout <<"Done." << std::endl;
    
}




