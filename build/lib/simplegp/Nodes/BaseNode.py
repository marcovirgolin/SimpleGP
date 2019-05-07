import numpy as np


class Node:	# Base class with general functionalities

	def __init__(self):
		self.fitness = np.inf
		self.parent = None
		self.arity = 0	# arity is the number of expected inputs
		self._children = []

	def GetSubtree( self ):
		result = []
		self.__GetSubtreeRecursive(result)
		return result

	def AppendChild( self, N ):
		self._children.append(N)
		N.parent = self

	def DetachChild( self, N ):
		assert(N in self._children)
		for i, c in enumerate(self._children):
			if c == N:
				self._children.pop(i)
				N.parent = None
				break
		return i

	def InsertChildAtPosition( self, i, N ):
		self._children.insert( i, N )
		N.parent = self

	def GetOutput( self, X ):
		return None

	def GetDepth(self):
		n = self
		d = 0
		while (n.parent):
			d = d+1
			n = n.parent
		return d

	def __GetSubtreeRecursive( self, result ):
		result.append(self)
		for c in self._children:
			c.__GetSubtreeRecursive( result )
		return result

