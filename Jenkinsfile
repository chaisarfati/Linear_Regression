pipeline {
  
  agent any
  
  stages {
    
    stage("init") {
      echo "Pipeline triggered on date ${TAG_DATE} (UNIX TIMESTAMP : ${TAG_UNIXTIME})" 
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
