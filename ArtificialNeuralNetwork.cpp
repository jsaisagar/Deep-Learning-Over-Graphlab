/**
 *
 * Authored by Jinka Sai Sagar
 *
 * This program illustrates the API I have designed for using Deep learning over
 * graphlab. I solved the usecase of Hand Writing Recognition (MNIST digits). I am also able to solve
 * Restricted Boltzmann machines(RBM) using this. 
 *
 * I have changed the graph structure, Processing of Graphlab Engine and designed new API for programming on layers.
 * 
 **/



#include <graphlab/vertex_program/ilayer_program.hpp>
#include <graphlab/vertex_program/ivertex_program_edited.hpp>
#include <graphlab/engine/async_consistent_engine_edited.hpp>
#include <graphlab/graph/distributed_graph_edited.hpp>
#include <string.h>
#include <vector>
#include <fstream>

#include <Eigen/Dense>

#define MAX_LAYERS 2  

typedef Eigen::MatrixXd Matrix;
typedef Eigen::VectorXd Vector;
	
struct node {
	int counter;
	node():counter(0){}
	 void save(graphlab::oarchive& oarc) const {
		oarc << counter;
	 }
void load(graphlab::iarchive& iarc) {
		iarc >> counter;
}
};

typedef graphlab::distributed_graph_edited<node, float, node> graph_type;

/*Since many of deep learning algorithms are sequential. There are two APIs designed in layer program.
 * One for forward direction and another for Backward direction. Layer can communicate with the vertices via
 * context.
 */
class neural_layer_program :
        public graphlab::ilayer_program<graph_type, float>,
	public graphlab::IS_POD_TYPE {
	public:
	void positive(icontext_edit_type& context,
                                        layer_type& layer,
					layer_type& prev_layer) 
        {
		size_t a = layer.in_this();
		Matrix mat = layer.weight();

		if(prev_layer.id()== 10)
		{
	 		Vector input_to_this = prev_layer.input();
			++layer.data().counter;
	                context.execute(a, mat, input_to_this, layer.lid);

		}
		else{
			Vector input_to_this = prev_layer.output();
			//std::cout<<"Layer matrix is " <<mat<<std::endl;
			++layer.data().counter;
        	        context.execute(a, mat, input_to_this, layer.lid);

		}


	}
	
	void reverse(icontext_edit_type& context, layer_type& layer,
				layer_type& prev_layer, const Vector& output)
	{
		
		if(layer.lid = MAX_LAYERS)
		{
			Vector error = layer.error();
			std::cout<<"Error \n " << error.size() << std::endl;
		        Vector sigmoid = layer.output();
			for(int i =0; i<sigmoid.size(); i++)
			{
				sigmoid(i) = (sigmoid(i))*(1-sigmoid(i));
			}
			Vector put = layer.weightedInputs()-output;
			error = (sigmoid).array()*put.array();
			Vector p(10);
			layer.setError(&error);
		}
			Matrix weight_next = layer.weight();
			std::cout << "Weight_next is " << weight_next.transpose() <<std::endl;
			Vector error_next = layer.error();
			std::cout << " Error is \n" << error_next << std::endl; 
			Vector sigmoid = (prev_layer.output());
			for(int i =0; i<sigmoid.size(); i++)
                        {
                                sigmoid(i) = (sigmoid(i))*(1-sigmoid(i));
                        }

			Vector error_previous = (weight_next.transpose())*(error_next);	
			//std::cout << "Weight matrix is multiplied with error received to the layer \n" << error_previous <<std::endl;
			Vector coeff = sigmoid.array()*error_previous.array();
			prev_layer.setError(&coeff);
			//std::cout <<"Error at the first layer is \n"<<coeff << std::endl;
		//}	
			Matrix weight_change = (layer.error())*((prev_layer.output()).transpose());
			//std::cout << "Change in the weights for a single dataset is captured \n" << weight_change <<std::endl;
			//std::cout << " While the original weight matrix is \n" << layer.weight() << std::endl;
			weight_change = (weight_next) - 0.1*(weight_change);
			layer.setWeights(&weight_change);
	}		
		
};	
		
		 
class neural_vertex_program :
   public graphlab::ivertex_program_edited<graph_type, float, Matrix>,
	 public graphlab::IS_POD_TYPE{		
	 
	 private:
//        int counter;

	 public:
                const static double LEARNING_RATE = 0.01;
                static double layer;
	 //private:
	//int counter;
        //The node is switched on with this probability
         float probability;

	 	edge_dir_type gather_edges(icontext_type& context,
                             const vertex_type& vertex) const
       		 {
                	return graphlab::NO_EDGES;
        	 }
		void applyAnother(icontext_type& context,layer_id_type lid, vertex_type& vertex,
                        Matrix& mat, Matrix& input)
        {
                Vector vec = input;
		Vector weight = mat.transpose();
		//std::cout << "The matrix is \n" << weight.size() << "and the input is " << input.size() << std::endl; 
		//VERTEX HAS ACCESS TO LAYER DATA. HERE WE CAN UPDATE EVERYTHING ON THE LAYER DATA.
                int counter = vertex.get_layer_data(lid).counter;
			//std::cout << "Vertex has counter value " <<vertex.data().counter << std::endl;
			double dot = vec.dot(weight);
			//std::cout << "The dot product of the first phase is " << dot  <<std::endl;
			double sigmoid = 1/(1+exp(-dot));
			vertex.send_weighted_inputs(lid,dot);
			//std::cout << "The sigmoid of the first phase is " << sigmoid  <<std::endl;
			//if(sigmoid>0.4)
			
			vertex.send_sigmoid(lid,sigmoid);

}
};


int main(int argc, char** argv) {
	graphlab::mpi_tools::init(argc, argv);
	graphlab::distributed_control dc;
	graph_type graph(dc);
	/*ADDING A LAYER IN THE GRAPH CONSTRUCTS THE LAYER OBJECT WITH FOLLOWING FIELDS
	 * 1. iD
	 * 2. NUMBER OF NODES IN THE THIS LAYER
	 * 3. NUMBER OF NODES IN THE PREVIOUS LAYER
	 * 4. NUMBER OF NODES IN THE NEXT LAYER
	 * 5. LAYER DATA 
	*/
	graph.add_layer(10,784,0,15,node());
	for(int i = 0; i<785; i++)
	{
		graph.add_vertex((i),node());
	}
	//add_layer(id,num_in_this,num_prev, num_next, ldata)
	graph.add_layer(20,15,784,4,node());
	graph.add_layer(30,10,15,0,node());
	graph.finalize();
	//graph.add_layer(1,2,3,4,'a');
	graph.layer_finalized();
	//graphlab::omni_engine<pagerank_program> engine(dc, graph, "async");
	graphlab::async_consistent_engine_edited<neural_vertex_program,neural_layer_program> engine(dc, graph);
	Eigen::VectorXd v(10);
	v << 1,0,1,1,1,1,0,0,0,0;
	Eigen::VectorXd k(784);
	 Eigen::VectorXd kt(784);
	std::cout<<"Weight Matrix before starting \n" << engine.returnWeight()<<std::endl;
	//std::cout<<"Befr parsing " <<std::endl;
 	std::ifstream file("sample.csv");
	std::ofstream files;
	std::string s;
	std::vector<std::string> st;       
	//std::cout<<"After initializing file parsing " <<std::endl;
	files.open("output.csv");
	int jp = 0;
	while((file.good())&&(jp<60000))
	{
		getline(file,s);
		st.push_back(s);
		jp++;
		for(int i =0; i<784; i++)
	        {
       		        k(i) = s[i]-48;
        	}
		int expected_out = s[784]-48;
		engine.initialize_input(&k);
		engine.run(3);
		Eigen::VectorXd exp_out(10);
		exp_out << 0,0,0,0,0,0,0,0,0,0;
		if(expected_out)
		exp_out(expected_out-1) = 1;
		else
		 exp_out(expected_out) = 1;

		engine.backPropagation(3,exp_out);
		files <<"Output of the data with index "<<jp<<"\n" << engine.returnOutput() <<std::endl;
		}
	files.close();
	 std::cout<<"Weight Matrix after running the engine \n" << engine.returnWeight()<<std::endl;
	std::ifstream testFile("test_modified.csv");
        std::string t;
        std::vector<std::string> tt;
	jp = 0;

	while((testFile.good())&&(jp<3))
	{
		getline(testFile,t);
		tt.push_back(t);
		jp++;
		for(int i =0; i<784; i++)
                {
                        kt(i) = s[i]-48;
		}
		engine.initialize_input(&kt);
		engine.run(3);
		//std::cout<< " Output from test \n" <<engine.returnOutput()<<std::endl;

	//	std::cout<< " Test Vector \n" <<k<<std::endl;
	}
	graphlab::mpi_tools::finalize();
	return 0;
}
