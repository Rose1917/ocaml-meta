#### what is the advantage of the meta-operator 
* The user-customized operator can also implement the auto-gradient:like owl,if you want to use the auto-gradient feature, you have to use the operator owl provides

#### what does a deeplearning framework need to provide
* easy to use
* effiency

#### why we need to use the ref type of the gradient
* to avoid the gradient-confusion:if there are one more path to the input variable,then the gradient confusion happends

#### what is the function of the adj_function
* to update the work_stack

#### what is the relationship between the owl and this project
* we use the basic data type and functions based on the data type

#### how to avoid retrained in the back-propogation
* set the if_trained value to the value 

#### any further work?
* add more datatypes supported
* to replace the owl dependency to improve the effiency( maybe it will work)
* to add specified Matrix Vector support 

