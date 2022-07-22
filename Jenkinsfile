pipeline {
  
  agent any
  
  
  parameters {
   choice(name: 'VERSION', choices: ['1.0.0','2.0.0','3.0.0'] , description:"version to deploy on prod") 
   booleanParam(name: 'executeTests', defaultValue: true, description:'')
  }
  
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
        echo "building app with version ${VERSION}" 
      }
    }
    
    stage("test"){
      
      when {
        expression {
         params.executeTests == true
        }
      }
      
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
