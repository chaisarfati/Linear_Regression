pipeline {
  
  agent any
  
  stages {
    
    stage("init") {
      steps {
        echo "Pipeline triggered"
        sh "ls"
      }
    }
    
    stage("build"){
      
      when {
        expression {
         GIT_BRANCH == "develop"
        }
      }
      
      steps {
        echo 'building app' 
      }
    }
    
    stage("test"){
      steps {
        echo 'testing app' 
      }
    }
    
    stage("deploy"){
      steps {
        echo 'deploying app' 
      }
    }
    
  }
  
  post {
   
    always {
     echo "Done" 
    }
    
    success {
     echo "It is a success" 
    }
    
        
    failure {
     echo "It is a failure" 
    }
    
  }
  
  
}
