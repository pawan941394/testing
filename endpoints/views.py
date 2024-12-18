
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from . import llm  # Import the LLM workflow code

# Initialize chat history
chat_history = {}

class LLMEndpoint(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, format=None):
        try:
            user_input = request.data.get('input')
            username = request.data.get('username', 'default_user')

            # Process the user input
            response = llm.process_user_input(user_input, chat_history, username)

            return Response({"status": "success", "response": response}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"status": "error", "message": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
