package hw1;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.PriorityQueue;
import java.util.Random;
import java.util.stream.Collectors;

public class EightPuzzleTest {
	private List<Integer> puzzle;
	
	private List<Integer> goalState = List.of(0,1,2,3,4,5,6,7,8);	
	
	private int maxNodes = Integer.MAX_VALUE; 
	
	private int cost = 0; 
			
	//private helper method which returns the index of the blank tile 
	private int returnBlankPuzzleIndex() {
		return puzzle.indexOf(0); 
	}
	
	//sets the state of the puzzle 
	public void setState(List<Integer> puzzle) {
		this.puzzle = puzzle; 
	}

	//prints the state of the puzzle 
	public String printState() {
		int index = 1; 
		StringBuilder sb = new StringBuilder(); 
		for(int i=0; i<9; i++) {
			sb.append(this.puzzle.get(i));
			if(index %3 == 0) {
				sb.append("\n"); 
			}
			index++; 
		}
		System.out.println("This is the current state" + "\n" + sb.toString()); 
		return sb.toString(); 
	}
	
	//converts the integer value of the puzzle to the 3x3 string puzzle 
	private String puzzleString(List<Integer> parent) {
		int index = 1; 
		StringBuilder sb = new StringBuilder(); 
		for(int i=0; i<9; i++) {
			sb.append(parent.get(i));
			if(index %3 == 0) {
				sb.append("\n"); 
			}
			index++; 
		}
		return sb.toString(); 
	}
	//converts the 3x3 string puzzle to the list<integer> puzzle 
	private List<Integer> puzzleInteger(String parent) {
		List<Integer> returnList = new ArrayList<Integer>();
		for(int i=0; i<parent.length(); i++) {
			if(Character.getNumericValue(parent.charAt(i)) != -1) {
			returnList.add(Character.getNumericValue(parent.charAt(i))); 
			}
		}
		return returnList; 
	}

	//move function 
	public boolean move(String direction) {
		if(direction.equals("up")){
			return this.moveUp(); 
		}
		else if(direction.equals("down")){
			return this.moveDown(); 
		}
		else if(direction.equals("left")){
			return this.moveLeft(); 
		}
		else if(direction.equals("right")){
			return this.moveRight(); 
		}
		else {
			return false; 
		}
	}
	
	//private helper method for moving up
	private boolean moveUp() {
		Integer blankIndex = returnBlankPuzzleIndex(); 
		if( blankIndex -3 < 0) {
			return false; 
		}
		Integer temp = this.puzzle.get(blankIndex - 3); 
		this.puzzle.set(blankIndex-3, 0); 
		this.puzzle.set(blankIndex, temp); 
		return true;	
	}
	
	//private helper method for moving down 
	public boolean moveDown() {
		Integer blankIndex = returnBlankPuzzleIndex(); 
		if( blankIndex + 3 > 8) {
			return false; 
		}
		Integer temp = this.puzzle.get(blankIndex + 3); 
		this.puzzle.set(blankIndex+3, 0); 
		this.puzzle.set(blankIndex, temp); 
		return true;	
	}
	
	//private helper method for moving left 
	public boolean moveLeft() {
		Integer blankIndex = returnBlankPuzzleIndex(); 
		if( blankIndex % 3 == 0) {
			return false; 
		}
		Integer temp = this.puzzle.get(blankIndex - 1); 
		this.puzzle.set(blankIndex-1, 0); 
		this.puzzle.set(blankIndex, temp); 
		return true;	
	}

	//private helper method for moving right 
	public boolean moveRight() {
		Integer blankIndex = returnBlankPuzzleIndex(); 
		if( blankIndex % 3 == 2) {
			return false; 
		}
		Integer temp = this.puzzle.get(blankIndex + 1); 
		this.puzzle.set(blankIndex + 1, 0); 
		this.puzzle.set(blankIndex, temp); 
		return true;
	}
	//sets the maximum number of nodes 
	public void maxNodes(int n) {
		if(n<0) {
			throw new IllegalArgumentException(); 
		}
		this.maxNodes = n; 
	}
	
	//private helper method to find h1value 
	private int h1Value(List<Integer> parent) {
		int misplacedTiles = 0; 
		for(int i=0; i<parent.size(); i++) {
			if(parent.get(i) != goalState.get(i)) {
				misplacedTiles++; 
			}
		}
		//calculates the number of misplaced tiles 
		return misplacedTiles; 
		
	}
	
	//randomizes state 
	public void randomizeState(int n) {
		Random random = new Random();
		//create seed 
		random.setSeed(10);
		setState(new ArrayList<>(this.goalState));
		boolean foundState = false; 
		int possibility; 
		for(int i=1; i<=n; i++) {
		while(!foundState) {
			possibility = random.nextInt(4)+1; 
			if(possibility == 1) {
				boolean valueUp = this.moveUp(); 
				if(valueUp) {
					foundState = true; 
					break; 
				}
			}
			else if(possibility == 2) {
				boolean valueDown = this.moveDown(); 
				if(valueDown) {
					foundState = true; 
					break; 
				}
			}
			else if(possibility == 3) {
				boolean valueLeft = this.moveLeft(); 
				if(valueLeft) {
					foundState = true; 
					break; 
				}
			}
			else {
				boolean valueRight = this.moveRight(); 
				if(valueRight) {
					foundState = true; 
					break; 
				}
			}
			}
		foundState = false; 
		}	
	}
	
	//private helper method to expand parent and find its children 
	private List<String> expand(List<Integer> parent){
		List<String> childrenList = new ArrayList<String>(); 
		//finds the 0 index
		int zero_index = parent.indexOf(0);  
		//move up 
		if(zero_index -3 >= 0) {
			//deep copy of parent 
			List<Integer> parentCopy = new ArrayList<Integer>(parent); 
			Integer temp = parentCopy.get(zero_index - 3); 
			parentCopy.set(zero_index-3, 0); 
			parentCopy.set(zero_index, temp); 
			//adds child to list 
			childrenList.add(puzzleString(parentCopy)); 
		}
		//move down 
		if(zero_index + 3 < 9) {
			//deep copy of parent 
			List<Integer> parentCopy = new ArrayList<Integer>(parent);
			Integer temp = parentCopy.get(zero_index + 3); 
			parentCopy.set(zero_index+3, 0); 
			parentCopy.set(zero_index, temp); 
			//adds child to list 
			childrenList.add(puzzleString(parentCopy)); 
		}
		//move left 
		if(zero_index % 3 != 0 ) {
			//deep copy of parent 
			List<Integer> parentCopy = new ArrayList<Integer>(parent);
			Integer temp = parentCopy.get(zero_index - 1); 
			parentCopy.set(zero_index-1, 0); 
			parentCopy.set(zero_index, temp); 
			//adds child to list 
			childrenList.add(puzzleString(parentCopy)); 
		}
		//move right
		if(zero_index %3 !=2) {
			//deep copy of parent 
			List<Integer> parentCopy = new ArrayList<Integer>(parent);
			Integer temp = parentCopy.get(zero_index + 1); 
			parentCopy.set(zero_index + 1, 0); 
			parentCopy.set(zero_index, temp); 
			//adds child to list
			childrenList.add(puzzleString(parentCopy)); 
		}
		//returns all children 
		return childrenList; 
	}
	
	private void solve_h1() {
		//resets cost to 0 
		this.cost= 0; 
		//set parent as initial state of puzzle 
		PriorityQueue<List<String>> frontier = new PriorityQueue<>(Comparator.comparingInt(list -> Integer.parseInt(list.get(0))));
		List<Integer> parent = this.puzzle; 
		//first element is cost + heuristic value, cost, and then the current puzzle state 
		String goalState = puzzleString(this.goalState);
 		List<String> frontierFirstValue = List.of(String.valueOf(this.cost + h1Value(this.puzzle)), puzzleString(parent));  
 		frontier.add(frontierFirstValue); 
 		//hash map of reached states 
 		HashSet<List<Integer>> reachedStates = new HashSet<List<Integer>>(); 
 		//hash map of parent and child to retrace path
 	    HashMap<String, String> cameFrom = new HashMap<>();
 	    cameFrom.put(puzzleString(parent), "Start");
 		//adds parent to reachedStates map 
 		reachedStates.add(parent); 
 		//boolean to keep track of whether goal state has been reached or not 
		boolean hasReachedGoalState = false; 
		int moves = 0; 
		if(parent.equals(this.goalState)) {
			hasReachedGoalState = true; 
		}
		//need to keep track of whether queue is not empty, number of nodes explored within max nodes explored, goal state not reached
		while(!frontier.isEmpty() && (reachedStates.size() < this.maxNodes) && (hasReachedGoalState == false)) {
			List<String> queueValue = frontier.poll();
			String poll = queueValue.get(1);
			if(poll.equals(goalState)) {
				hasReachedGoalState = true; 
				List<String> path = new ArrayList<>();
				String current = goalState;
				this.setState(puzzleInteger(poll)); 
				//adds moves to path 
				while (!current.equals("Start")) {
				    path.add(current);
				    current = cameFrom.get(current);
				}
				System.out.println("This is the path of h1"); 
				// Print the sequence of moves in reverse order
				for (int i = path.size() - 1; i >= 0; i--) {
				    System.out.println(path.get(i));
				    moves++;
				}
				break; 
			}
			this.cost++; 
			//expands list of childen 
			List<String> listChildren = expand(puzzleInteger(poll));
			for(int i = 0; i<listChildren.size(); i++) {
				String childString = listChildren.get(i); 
				List<Integer> child = puzzleInteger(childString); 
				//if child not in reached states
				if(reachedStates.contains(child)==false){
					int h1Value = h1Value(child); 
					String heuristicValue = String.valueOf(this.cost + h1Value); 
					List<String> frontierValue = List.of(heuristicValue, childString); 
					//adds child to reached states, queue and path
			 		reachedStates.add(child); 
			 		frontier.add(frontierValue); 
	                cameFrom.put(childString, poll);
				}
			}
		}
        if (hasReachedGoalState == true) {
            System.out.println("Number of nodes:" + reachedStates.size()); 
            System.out.println("Number of moves: " + moves); 
        }
        else {
           System.out.println("No solution found for A-star h1. "); 
        }
	}
	
	//private helper method for finding the h2 value given parent for Manhattan distance 
	private int h2Value(List<Integer> parent) {
		int heuristicValue = 0; 
		for(int i=0; i<parent.size(); i++) {
			int goalStateIndex = goalState.indexOf(parent.get(i)); 
			if(parent.get(i) != goalState.get(i)) {
				heuristicValue = heuristicValue + Math.abs(i-goalStateIndex) / 3  + Math.abs(i - goalStateIndex) % 3; 
			}
		}
		return heuristicValue; 	
	}

	//private helper method to solve h2
	private void solve_h2() {
		this.cost = 0; 
		//set parent as initial state of puzzle 
		PriorityQueue<List<String>> frontier = new PriorityQueue<>(Comparator.comparingInt(list -> Integer.parseInt(list.get(0))));
		List<Integer> parent = this.puzzle; 
		//first element is cost + heuristic value, cost, and then the current puzzle state 
		String goalState = puzzleString(this.goalState);
 		List<String> frontierFirstValue = List.of(String.valueOf(this.cost + h2Value(this.puzzle)), puzzleString(parent));  
 		frontier.add(frontierFirstValue); 
 		//hash map for reached states 
 		HashSet<List<Integer>> reachedStates = new HashSet<List<Integer>>(); 
 		//hash map for path 
 	    HashMap<String, String> cameFrom = new HashMap<>();
 	    cameFrom.put(puzzleString(parent), "Start");
 		//adds parent to reachedStates map 
 		reachedStates.add(parent); 
 		//boolean to keep track of whether goal state has been reached or not 
		boolean hasReachedGoalState = false; 
		int moves = 0; 
		if(parent.equals(this.goalState)) {
			hasReachedGoalState = true; 
		}
		//need to keep track of whether queue is not empty, number of nodes explored within max nodes explored, goal state not reached
		while(!frontier.isEmpty() && (reachedStates.size() < this.maxNodes) && (hasReachedGoalState == false)) {
			List<String> queueValue = frontier.poll();
			String poll = queueValue.get(1);
			if(poll.equals(goalState)) {
				hasReachedGoalState = true; 
				List<String> path = new ArrayList<>();
				String current = goalState;
				this.setState(puzzleInteger(poll)); 
				while (!current.equals("Start")) {
				    path.add(current);
				    current = cameFrom.get(current); 
				}
				System.out.println("This is the path of h2"); 
				// Print the sequence of moves in reverse order
				for (int i = path.size() - 1; i >= 0; i--) {
				    System.out.println(path.get(i));
				    moves++;
				} 
				break; 
			}
			this.cost++; 
			//expands current state 
			List<String> listChildren = expand(puzzleInteger(poll));
			for(int i = 0; i<listChildren.size(); i++) {
				String childString = listChildren.get(i); 
				List<Integer> child = puzzleInteger(childString); 
				//if child not in reached states 
				if(reachedStates.contains(child)==false){
					int h2Value = h2Value(child); 
					String heuristicValue = String.valueOf(this.cost + h2Value); 
					List<String> frontierValue = List.of(heuristicValue, childString); 
					//adds to reached states, queue, path 
			 		reachedStates.add(child); 
			 		frontier.add(frontierValue); 
	                cameFrom.put(childString, poll);
				}
			}
		}
        if (hasReachedGoalState == true) {
            System.out.println("Number of nodes:" + reachedStates.size()); 
            System.out.println("Number of moves: " + moves); 
        }
        else {
           System.out.println("No solution found for A-star h2. "); 
        }
	}
	
	//public method to solve A star 
	public void solveAStar(String heuristic) {
		if(heuristic.equals("h1")) {
			this.solve_h1();
		}
		if(heuristic.equals("h2")) {
			this.solve_h2();
		}
	}
	
	//public method to solve beam 
	public void solveBeam(int k) {
		//resets cost to 0 
		this.cost = 0;
		//set parent as initial state of puzzle 
		PriorityQueue<List<String>> frontier = new PriorityQueue<>(Comparator.comparingInt(list -> Integer.parseInt(list.get(0))));
		List<Integer> parent = this.puzzle; 
		//first element is cost + heuristic value, cost, and then the current puzzle state 
		String goalState = puzzleString(this.goalState);
 		List<String> frontierFirstValue = List.of(String.valueOf(this.cost + h2Value(this.puzzle)), puzzleString(parent));  
 		frontier.add(frontierFirstValue); 
 		//hash map keep track of the reached states
 		HashSet<List<Integer>> reachedStates = new HashSet<List<Integer>>(); 
 	    HashMap<String, String> cameFrom = new HashMap<>();
 	    cameFrom.put(puzzleString(parent), "Start");
 		//adds parent to reachedStates map 
 		reachedStates.add(parent); 
 		//boolean to keep track of whether goal state has been reached or not 
		boolean hasReachedGoalState = false; 
		int moves = 0; 
		if(parent.equals(this.goalState)) {
			hasReachedGoalState = true; 
		}
		//need to keep track of whether queue is not empty, number of nodes explored within max nodes explored, goal state not reached
		while(!frontier.isEmpty() && (reachedStates.size() < this.maxNodes) && (hasReachedGoalState == false)) {
			List<String> queueValue = frontier.poll();
			String poll = queueValue.get(1);
			if(poll.equals(goalState)) {
				hasReachedGoalState = true; 
				this.setState(puzzleInteger(poll)); 
				List<String> path = new ArrayList<>();
				String current = goalState;
				//finds current path 
				while (!current.equals("Start")) {
				    path.add(current);
				    current = cameFrom.get(current); 
				}
				System.out.println("This is the path of beam"); 
				// Print the sequence of moves in reverse order
				for (int i = path.size() - 1; i >= 0; i--) {
				    System.out.println(path.get(i));
				    moves++;
				} 
				break; 
			}
			this.cost++; 
			//expands children of current state 
			List<String> listChildren = expand(puzzleInteger(poll));
			for(int i = 0; i<listChildren.size(); i++) {
				String childString = listChildren.get(i); 
				List<Integer> child = puzzleInteger(childString); 
				//checks if reached states contains child 
				if(reachedStates.contains(child)==false){
					int h2Value = h2Value(child); 
					String heuristicValue = String.valueOf(this.cost + h2Value); 
					List<String> frontierValue = List.of(heuristicValue, childString); 
					//adds child to queue, reached states, and path 
			 		reachedStates.add(child); 
			 		frontier.add(frontierValue); 
	                cameFrom.put(childString, poll);
				}
			}
			//if k value is less than frontier size then create new priority queue of size k 
			if(k<frontier.size()) {
				PriorityQueue<List<String>> bestKStates = new PriorityQueue<>(Comparator.comparingInt(list -> Integer.parseInt(list.get(0))));
				for(int i=0; i<k; i++) {
					List<String> bestState = frontier.poll();
					bestKStates.add(bestState); 
					}
				frontier = bestKStates; 
			}	
			
		}
        if (hasReachedGoalState == true) {
            System.out.println("Number of nodes:" + reachedStates.size()); 
            System.out.println("Number of moves: " + moves); 
        }
        else {
           System.out.println("No solution found for beam search. "); 
        }
	}
	
	//main method which reads from file 
	public static void main(String[] args) throws IOException {
		EightPuzzleTest eightPuzzle = new EightPuzzleTest(); 
		File file = new File("/Users/arohimehta/eclipse-workspace/IntroAI/src/hw1/eightPuzzleTest");
		try {
			BufferedReader br = new BufferedReader(new FileReader(file));
			String st;
			String[] command = null; 
			//iterates through each line 
			while ((st = br.readLine()) != null) {
				command = st.split(" "); 
				//if set state 
				if (command[0].equals("setState")) {
					List<Integer> puzzle = new ArrayList<Integer>();
					for(int i = 1; i<4; i++) {
					command[i].chars().map(Character::getNumericValue).forEach(puzzle::add); 
					}
					eightPuzzle.setState(puzzle);
				}
				//if print state 
				if (command[0].equals("printState")) {
					eightPuzzle.printState(); 
				}
				//if max nodes 
				if(command[0].equals("maxNodes")) {
					eightPuzzle.maxNodes(Integer.parseInt(command[1])); 
				}
				//if move 
				if (command[0].equals("move")) {
					eightPuzzle.move(Objects.requireNonNull(command[1])); 
				}
				//if randomize state 
				if(command[0].equals("randomizeState")) {
					eightPuzzle.randomizeState(Integer.parseInt(Objects.requireNonNull(command[1]))); 
				}
				//if solve a star or beam 
				if(command[0].equals("solve")){
					if(command[1].equals("A-star")) {
						eightPuzzle.solveAStar(Objects.requireNonNull(command[2]));
						}
					if(command[1].equals("beam")) {
						eightPuzzle.solveBeam(Integer.parseInt(Objects.requireNonNull(command[2]))); 
					}
				}

			}
		}
		catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
}
